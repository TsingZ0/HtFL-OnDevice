#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Expected 2 arguments"
    echo "Usage: $0 <DATASET_DIR> <ASSIGNMENT_STRATEGY>"
    echo "ASSIGNMENT_STRATEGY can be 'identify' or 'random'"
    exit 1
fi

if [ -z "$COLEXT_CLIENT_ID" ]; then
    echo "Error: COLEXT_CLIENT_ID environment variable is not set. Are you running this script on a CoLExT env?"
    exit 1
fi

DATASET_DIR=$1
ASSIGNMENT_STRATEGY=$2

case $ASSIGNMENT_STRATEGY in
    "identity")
        PARTITION_ID=$COLEXT_CLIENT_ID
        ;;
    "random")
        PARTITION_ID=$(($RANDOM % $(ls $DATASET_DIR/train | wc -l)))
        ;;
    *)
        echo "Invalid assignment strategy: $ASSIGNMENT_STRATEGY"
        exit 1
        ;;
esac

# Create symlink to train and test data
DATA_DIR="device_data"
mkdir -p $DATA_DIR
echo "Assigning partition $PARTITION_ID to this device"
ln -sf $DATASET_DIR/train/$PARTITION_ID.npz $DATA_DIR/train_${COLEXT_CLIENT_ID}.npz
ln -sf $DATASET_DIR/test/$PARTITION_ID.npz  $DATA_DIR/test_${COLEXT_CLIENT_ID}.npz
