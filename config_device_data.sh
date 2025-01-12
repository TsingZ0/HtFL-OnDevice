#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Expected 2 arguments"
    echo "Usage: $0 <DATASET_DIR> <PARTITION_ID>"
    exit 1
fi

DATASET_DIR=$1
PARTITION_ID=$2

# Create symlink to train and test data
DATA_DIR="device_data"
mkdir -p $DATA_DIR
echo "Assigning partition $PARTITION_ID to this device"
ln -sf $DATASET_DIR/train/$PARTITION_ID.npz $DATA_DIR/train.npz
ln -sf $DATASET_DIR/test/$PARTITION_ID.npz  $DATA_DIR/test.npz
