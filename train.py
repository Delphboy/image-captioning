import argparse
import itertools
import multiprocessing
import os
import random
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing
from tqdm import tqdm

import evaluation
from evaluation import Cider, PTBTokenizer
from utils import factories
from utils.model_storage import load_model, save_model
from utils.plots import plot_training_charts

import wandb

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print("Setting DEVICE to:", DEVICE)


def train_epoch_xe(
    model,
    dataloader,
    loss_fn,
    optim,
    scheduler,
    epoch,
    vocab_size,
):
    model.train()
    running_loss = 0.0

    desc = "Epoch %d - train" % epoch

    with tqdm(desc=desc, unit="it", total=len(dataloader)) as pbar:
        for it, (input_features, targets, _) in enumerate(dataloader):
            optim.zero_grad()
            input_features = input_features.to(DEVICE)
            targets = targets.to(DEVICE)

            out = model(input_features, targets)
            captions_gt = targets[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, vocab_size), captions_gt.view(-1))

            loss.backward(loss)

            optim.step()

            this_loss = loss.item()
            running_loss += this_loss

            wandb.log({"train_loss": this_loss})

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    scheduler.step()
    loss = running_loss / (it + 1)
    return loss


def train_epoch_scst(model, dataloader, optim, cider, epoch, vocab, beam_size=3):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = 0.0
    running_reward_baseline = 0.0
    model.train()
    running_loss = 0.0
    seq_len = 20

    desc = "Epoch %d - train" % epoch
    with tqdm(desc=desc, unit="it", total=len(dataloader)) as pbar:
        for it, (detections, _, caps_gt) in enumerate(dataloader):
            detections = [det.to(DEVICE) for det in detections]
            outs, log_probs = model.beam_search(
                detections,
                seq_len,
                vocab.vocab.stoi["<eos>"],
                beam_size,
            )
            optim.zero_grad()

            # Rewards
            caps_gen = vocab.decode(outs.view(-1, seq_len))
            caps_gt = list(
                itertools.chain(
                    *(
                        [
                            c,
                        ]
                        * beam_size
                        for c in caps_gt
                    )
                )
            )
            caps_gen, caps_gt = tokenizer_pool.map(
                evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt]
            )
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = (
                torch.from_numpy(reward)
                .to(DEVICE)
                .view(detections[0].shape[0], beam_size)
            )
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(
                loss=running_loss / (it + 1),
                reward=running_reward / (it + 1),
                reward_baseline=running_reward_baseline / (it + 1),
            )
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


@torch.no_grad()
def evaluate_epoch_xe(model, dataloader, loss_fn, epoch):
    model.eval()
    running_loss = 0.0

    desc = "Epoch %d - validation" % epoch

    with tqdm(desc=desc, unit="it", total=len(dataloader)) as pbar:
        for it, (input_features, targets, _) in enumerate(dataloader):
            input_features = input_features.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(input_features, targets[:, :-1])

            loss = loss_fn(
                outputs.reshape(-1, outputs.shape[2]), targets[:, 1:].reshape(-1)
            )

            this_loss = loss.item()
            running_loss += this_loss

            wandb.log({"val_loss": this_loss})

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / (it + 1)
    return loss


@torch.no_grad()
def evaluate_metrics(
    model, dataloader, text_field, epoch, beam_width=5, is_test=False, max_len=30
):
    eos_index = text_field.vocab._stoi[text_field.vocab.get_special_token("eos_token")]
    model.eval()
    gen = {}
    gts = {}
    with tqdm(
        desc="Epoch %d - evaluation" % epoch, unit="it", total=len(dataloader)
    ) as pbar:
        for it, (input_features, _, caps_gt) in enumerate(iter(dataloader)):
            input_features = input_features.to(DEVICE)
            with torch.no_grad():
                out = model.beam_search(
                    visual=input_features,
                    max_len=max_len,
                    eos_idx=eos_index,
                    beam_size=beam_width,
                )
                caps_gen = text_field.decode(out, join_words=True)
            for i in range(len(caps_gt)):
                gen[f"{it}_{i}"] = [caps_gen[i]]
                gts[f"{it}_{i}"] = [caption for caption in caps_gt[i]]
            pbar.update()

    print("-" * 10)
    print(f"Predicted: {gen['0_0']}")
    print("Ground Truth:")
    print([f"{g}" for g in gts["0_0"]])
    print("-" * 10)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen, is_test=is_test)

    if is_test:
        test_metrics = {
            "BLEU-1": scores["BLEU"][0],
            "BLEU-2": scores["BLEU"][1],
            "BLEU-3": scores["BLEU"][2],
            "BLEU-4": scores["BLEU"][3],
            "METEOR": scores["METEOR"],
            "ROUGE": scores["ROUGE"],
            "CIDEr": scores["CIDEr"],
            "SPICE": scores["SPICE"],
        }
        data = [[k, v] for k, v in test_metrics.items()]
        table = wandb.Table(data=data, columns=["Metric", "Score"])
        wandb.log(
            {
                "Test Metrics": wandb.plot.bar(
                    table, "Metric", "Score", title="Test Metrics"
                )
            }
        )
    else:
        wandb.log(
            {
                "val/BLEU-1": scores["BLEU"][0],
                "val/BLEU-2": scores["BLEU"][1],
                "val/BLEU-3": scores["BLEU"][2],
                "val/BLEU-4": scores["BLEU"][3],
                "val/METEOR": scores["METEOR"],
                "val/ROUGE": scores["ROUGE"],
                "val/CIDEr": scores["CIDEr"],
            }
        )

    return scores


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    # set up argument parser
    parser = argparse.ArgumentParser(description="Train model")

    # Dataset arguments
    parser.add_argument(
        "--captions_file",
        type=str,
        default=None,
        required=True,
        help="Path to the Karpathy JSON file",
    )
    parser.add_argument(
        "--butd_root",
        type=str,
        default=None,
        required=True,
        help="Path to the BUTD image features",
    )
    parser.add_argument(
        "--sgae_root",
        type=str,
        default=None,
        required=False,
        help="Path to the SGAE semantic graph data",
    )
    parser.add_argument(
        "--vsua_root",
        type=str,
        default=None,
        required=False,
        help="Path to the VSUA geometric graph data",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        required=True,
        help="Type of input to use",
        choices=["butd", "semantic_basic", "spatial_basic"],
    )
    parser.add_argument(
        "--feature_limit",
        type=int,
        default=50,
        help="How many features to use per image (default: 50)",
    )

    # General Model parameters
    parser.add_argument(
        "--token_dim", type=int, default=768, help="Dimension of the token space"
    )

    # Encoder Model Parameters
    parser.add_argument(
        "--enc_model_type",
        type=str,
        default="gcn",
        choices=["none", "pool", "gcn", "gin", "gat"],
        help="Type of encoder model to use",
    )
    parser.add_argument(
        "--enc_num_layers", type=int, default=2, help="Number of encoder layers"
    )

    # Decoder Model Parameters
    parser.add_argument(
        "--dec_lang_model",
        type=str,
        default="dual_lstm",
        help="Which language model to use",
        choices=["dual_lstm", "lstm"],
    )
    parser.add_argument(
        "--dec_num_layers",
        type=int,
        default=1,
        help="Number of layers to use in the language model",
    )

    # Training parameters
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument("--warm_up", type=int, default=1, help="Warm up steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="maximum epochs")
    parser.add_argument(
        "--force_rl_after", type=int, default=-1, help="force RL after (-1 to disable)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of pytorch dataloader workers"
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Random seed (-1) for no seed"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=-1,
        help="Patience for early stopping (-1 to disable)",
    )

    # Training options
    parser.add_argument("--resume_last", action="store_true")
    parser.add_argument("--resume_best", action="store_true")
    parser.add_argument(
        "--checkpoint_location",
        type=str,
        default="saved_models",
        help="Path to checkpoint save directory",
    )
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum decode length")

    # Regularisation and Normalisation settings
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    args = parser.parse_args()

    # Set random seed
    if args.seed != -1:
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # This is makes things deterministic
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not os.path.exists(args.checkpoint_location):
        os.makedirs(args.checkpoint_location)

    wandb_location = os.path.join(args.checkpoint_location, "wandb")
    if not os.path.exists(wandb_location):
        os.makedirs(wandb_location)

    wandb.init(
        project="image-captioning", name=args.exp_name, config=args, dir=wandb_location
    )

    # Load dataset
    train_data, val_data, test_data = factories.get_datasets(args)
    vocab = train_data.vocab
    vocab_size = len(vocab)
    train_dataloader = factories.get_dataloader(train_data, args)
    val_dataloader = factories.get_dataloader(
        val_data,
        args,
        shuffle=False,
    )
    test_dataloader = factories.get_dataloader(
        test_data,
        args,
        shuffle=False,
    )

    # SCST Things:
    scst_train_data, _, _ = factories.get_datasets(args)
    # TODO: Handle the different batch size needed for SCST
    scst_train_dataloader = factories.get_dataloader(scst_train_data, args)
    ref_caps_train = list(scst_train_data.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    # Load model
    model = factories.get_model(args, vocab).to(DEVICE)
    wandb.watch(
        model,
        criterion=nn.NLLLoss(
            ignore_index=vocab.stoi(vocab.get_special_token("pad_token"))
        ),
        log="all",
        log_freq=10,
    )

    # Setup optimiser and loss function
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 3, 0.8)

    loss_fn = nn.NLLLoss(ignore_index=vocab.stoi(vocab.get_special_token("pad_token")))

    use_rl = False
    best_cider = 0.0
    patience = 0
    epoch = 1
    training_losses = []
    training_scores = []
    validation_losses = []
    validation_scores = []

    if args.resume_last or args.resume_best:
        if args.resume_last:
            load_best = False
        else:
            load_best = True

        model, optim, scheduler, epoch, patience, best_cider, use_rl = load_model(
            model, optim, scheduler, args, load_best
        )
        print(f"Resuming from epoch {epoch} with best cider {best_cider}")

        if (not use_rl) and (epoch > args.force_rl_after and args.force_rl_after > 0):
            print("Switching to RL")
            use_rl = True
            optim = torch.optim.Adam(model.parameters(), lr=5e-6)

    # Training loop
    for epoch in range(epoch, args.epochs + 1):
        if use_rl:
            training_loss, reward, reward_baseline = train_epoch_scst(
                model, scst_train_dataloader, optim, cider_train, epoch, scst_train_data
            )
            print(
                f"Epoch {epoch} - train loss: {training_loss} - reward: {reward} - reward_baseline: {reward_baseline}"
            )
        else:
            training_loss = train_epoch_xe(
                model,
                train_dataloader,
                loss_fn,
                optim,
                scheduler,
                epoch,
                vocab_size,
            )
        training_losses.append(training_loss)

        # Validation
        with torch.no_grad():
            val_loss = evaluate_epoch_xe(model, val_dataloader, loss_fn, epoch)
            validation_losses.append(val_loss)
            scores = evaluate_metrics(
                model,
                val_dataloader,
                train_data,
                epoch,
                beam_width=args.beam_width,
                max_len=args.max_len,
            )
            validation_scores.append(scores)

        print(f"Epoch {epoch} - train loss: {training_loss}")
        print(f"Epoch {epoch} - validation loss: {val_loss}")
        print(f"Epoch {epoch} - validation scores: {scores}")

        cider = scores["CIDEr"]

        # Save model
        save_model(
            model,
            optim,
            scheduler,
            val_loss,
            cider,
            epoch,
            patience,
            best_cider,
            use_rl,
            args,
        )

        if cider > best_cider:
            best_cider = cider
            patience = 0
            print("Saving best model")
            last = f"{args.checkpoint_location}/{args.exp_name}_last.pth"
            best = f"{args.checkpoint_location}/{args.exp_name}_best.pth"
            copyfile(last, best)
        else:
            patience += 1

        if patience == args.patience or (epoch == args.force_rl_after and not use_rl):
            if not use_rl and args.force_rl_after > -1:
                print("Switching to RL")
                use_rl = True

                # load best model
                model, optim, epoch, patience, best_cider, use_rl = load_model(
                    model, optim, args, True
                )

                optim = torch.optim.Adam(model.parameters(), lr=5e-6)
            else:
                print("Early stopping")
                break
        print()

    # Load best model
    model, optim, epoch, patience, best_cider, use_rl = load_model(
        model, optim, args, True
    )

    # Evaluate on test set
    print("*" * 80)
    with torch.no_grad():
        scores = evaluate_metrics(
            model,
            test_dataloader,
            test_data,
            0,
            beam_width=args.beam_width,
            is_test=True,
            max_len=args.max_len,
        )
        print(f"Test scores: {scores}")
    print("*" * 80)

    wandb.finish()
    plot_training_charts(
        training_losses,
        validation_losses,
        validation_scores,
        args.exp_name,
    )
