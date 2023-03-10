#/bin/bash
mkdir /import/gameai-01/eey362/datasets/visual-genome
cd /import/gameai-01/eey362/datasets/visual-genome

# Download images
echo "Downloading images..."
echo "(This takes a while, go make a cuppa)"
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -O images_1.zip
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -O images_2.zip

echo "Images downloaded! Unzipping..."
unzip -d VG_100K images_1.zip
unzip -d VG_100K images_2.zip  

# Hacky as there are too many files
echo "Moving files..."
mv VG_100K/VG_100K/*0.jpg VG_100K/
mv VG_100K/VG_100K/*1.jpg VG_100K/
mv VG_100K/VG_100K/*2.jpg VG_100K/
mv VG_100K/VG_100K/*3.jpg VG_100K/
mv VG_100K/VG_100K/*4.jpg VG_100K/
mv VG_100K/VG_100K/*5.jpg VG_100K/
mv VG_100K/VG_100K/*6.jpg VG_100K/
mv VG_100K/VG_100K/*7.jpg VG_100K/
mv VG_100K/VG_100K/*8.jpg VG_100K/
mv VG_100K/VG_100K/*9.jpg VG_100K/
mv VG_100K/VG_100K/* VG_100K/

mv VG_100K/VG_100K_2/* VG_100K/

# Download image meta data
echo "Downloading image meta data..."
wget -c http://visualgenome.org/static/data/dataset/image_data.json.zip
unzip image_data.json.zip
rm image_data.json.zip

# Download region descriptions
echo "Downloading region descriptions..."
wget -c http://visualgenome.org/static/data/dataset/region_descriptions.json.zip
unzip region_descriptions.json.zip
rm region_descriptions.json.zip

# Download QA data
echo "Downloading QA data..."
wget -c http://visualgenome.org/static/data/dataset/question_answers.json.zip
unzip question_answers.json.zip
rm question_answers.json.zip

# Download objects
echo "Downloading objects..."
wget -c http://visualgenome.org/static/data/dataset/objects.json.zip
unzip objects.json.zip
rm objects.json.zip

# Download attributes
echo "Downloading attributes..."
wget -c http://visualgenome.org/static/data/dataset/attributes.json.zip
unzip attributes.json.zip
rm attributes.json.zip

# Download relationships
echo "Downloading relationships..."
wget -c http://visualgenome.org/static/data/dataset/relationships.json.zip
unzip relationships.json.zip
rm relationships.json.zip

# Download graphs
echo "Downloading graphs..."
wget -c http://visualgenome.org/static/data/dataset/region_graphs.json.zip
unzip region_graphs.json.zip
rm region_graphs.json.zip

wget -c http://visualgenome.org/static/data/dataset/scene_graphs.json.zip
unzip scene_graphs.json.zip
rm scene_graphs.json.zip

# Download mappings
echo "Downloading mappings..."
wget -c http://visualgenome.org/static/data/dataset/qa_to_region_mapping.json.zip
unzip qa_to_region_mapping.json.zip
rm qa_to_region_mapping.json.zip

# Download sysnet data
echo "Downloading synset data..."
wget -c http://visualgenome.org/static/data/dataset/synsets.json.zip
unzip synsets.json.zip
rm synsets.json.zip

wget -c http://visualgenome.org/static/data/dataset/object_synsets.json.zip
unzip object_synsets.json.zip
rm object_synsets.json.zip

wget -c http://visualgenome.org/static/data/dataset/attribute_synsets.json.zip
unzip attribute_synsets.json.zip
rm attribute_synsets.json.zip

wget -c http://visualgenome.org/static/data/dataset/relationship_synsets.json.zip
unzip relationship_synsets.json.zip
rm relationship_synsets.json.zip

echo "Done!"
