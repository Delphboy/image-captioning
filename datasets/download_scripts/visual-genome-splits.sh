cd /homes/hps01/image-captioning/datasets
wget -c https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/val.txt
sed -E -i 's/\.jpg xml\/[0-9]+\.xml//g' val.txt
sed -E -i 's/[A-Za-z_0-9]+\///g' val.txt

wget -c https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/test.txt
sed -E -i 's/\.jpg xml\/[0-9]+\.xml//g' test.txt
sed -E -i 's/[A-Za-z_0-9]+\///g' test.txt

wget -c https://raw.githubusercontent.com/peteanderson80/bottom-up-attention/master/data/genome/train.txt
sed -E -i 's/\.jpg xml\/[0-9]+\.xml//g' train.txt
sed -E -i 's/[A-Za-z_0-9]+\///g' train.txt