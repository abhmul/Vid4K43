TARGET_DIR="/media/abhmul/BackupSSD1/Datasets"
cd $TARGET_DIR
mkdir images
aws s3 --no-sign-request sync s3://open-images-dataset/train images/
aws s3 --no-sign-request sync s3://open-images-dataset/validation images/
aws s3 --no-sign-request sync s3://open-images-dataset/test images/