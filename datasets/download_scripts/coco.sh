#/bin/bash
mkdir coco
cd coco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

unzip train2014.zip
unzip val2014.zip
unzip test2014.zip

rm train2014.zip
rm val2014.zip
rm test2014.zip

cd ../
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotations_trainval2014.zip

rm annotations_trainval2014.zip

echo "COCO dataset downloaded and extracted."