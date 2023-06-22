import re
import boto3
from boto3.s3.transfer import TransferConfig
from io import BytesIO
import torch
from nemo.utils import AppState

def is_s3_path(filepath):
        return re.match("^s3://(.*)",filepath)

def download_s3_file_to_stream(s3_path: str, s3_client, chunk_size_MB: int = 64, max_concurrency: int = 15) -> BytesIO:
    bytes_buffer = BytesIO()
    bucket, key = parse_s3_url(s3_path)
    print(f"download s3 bucket {bucket}, key {key}")
    MB = 1024 ** 2 # Is this the correct megabyte calc for here?
    chunk_size = chunk_size_MB * MB
    config = TransferConfig(multipart_chunksize=chunk_size, max_concurrency=max_concurrency)
    s3_client.download_fileobj(bucket, key, bytes_buffer, Config=config)
    bytes_buffer.seek(0)
    return bytes_buffer

#TODO: This needs to handle s3 directories and file paths
def parse_s3_url(s3_path):
    s3_tokens = s3_path.split('/')
    bucket_name = s3_tokens[2]
    key = ""
    filename = s3_tokens[len(s3_tokens) - 1]
    if len(s3_tokens) > 4:
        for tokn in range(3, len(s3_tokens) - 1):
            key += s3_tokens[tokn] + "/"
        key += filename
    else:
        key += filename
    return bucket_name, key

def get_s3_file_names(s3_path,suffix):
    bucket_name, _ = parse_s3_url(s3_path)
    s3_client = boto3.client('s3')
    objs = s3_client.list_objects_v2(Bucket=bucket_name)['Contents']
    matched_files = []        
    for obj in objs:
        key = obj['Key']
        #timestamp = obj['LastModified']
        if key.endswith(suffix):              
            # Adding a new key value pair
            matched_files.append(key)   
    return matched_files

def s3_save(checkpoint,filepath):
        s3_client = boto3.client('s3')
        file_stream = BytesIO()
        torch.save(checkpoint, file_stream)
        file_stream.seek(0)
        MB = 1024 ** 2 # Is this the correct megabyte calc for here?
        config = TransferConfig(multipart_chunksize=128 * MB, max_concurrency=10)
        bucket, key = parse_s3_url(filepath)
        print(f"s3 save bucket {bucket}, key {key}")
        s3_client.upload_fileobj(file_stream, bucket, key, Config=config)

def load_s3_checkpoint(checkpoint_path):
        app_state = AppState()
        checkpoints = [None]
        # Loading checkpoint and broadcasting to other ranks
        s3_client = boto3.client('s3')
        file_stream: BytesIO = download_s3_file_to_stream(s3_path=checkpoint_path,s3_client=s3_client, chunk_size_MB=128, max_concurrency=15)
        
        #checkpoint = torch.load(file_stream)
        checkpoint = torch.load(file_stream,map_location=torch.device(f"cuda:{app_state.local_rank}"))
        #for x in checkpoint['state_dict'].items():
        #    print(f"state device {x[1].get_device()}, rank {app_state.global_rank}")
        #
        #for y in checkpoint['optimizer_states']:
        #    for z in y['state'].items():
        #        print(f"optimizer device {z[1]['exp_avg'].get_device()}, rank {app_state.global_rank}")
        #        #print(z[1]['exp_avg'])

        #checkpoints = [checkpoint]
        #logging.info('Broadcasting checkpoints to other ranks')
        # Check that this broadcast is allowed
        #dp_group = torch.distributed.get_process_group_ranks(app_state.data_parallel_group)
        #src = min(dp_group)
        #torch.distributed.broadcast_object_list(checkpoints, src=src, group=app_state.data_parallel_group)
        #return checkpoints[0]
        return checkpoint