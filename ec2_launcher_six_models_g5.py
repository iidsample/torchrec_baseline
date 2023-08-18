import sys
import boto3
import time

from pssh.clients import ParallelSSHClient

def return_args_trainers_dlrm(
    batch_size, private_ip_trainers, log_file_name
):
    """
    Arguments for trainers
    """

    run_args_trainers = list()
    for i in range(len(private_ip_trainers)):
        if i == 0:
            master_ip = "localhost"
        else:
            master_ip = private_ip_trainers[0]
        info = {"cmd": "rm -rf torchrec_baseline && git clone https://github.com/Jesse-Yan/torchrec_baseline.git && cd torchrec_baseline && bash run_trainers_dlrm.sh {} {} {} {}".format(batch_size, master_ip, len(private_ip_trainers), log_file_name)}
        run_args_trainers.append(info)
    return run_args_trainers


def return_args_trainers_wdn(
    batch_size, private_ip_trainers, log_file_name
):
    """
    Arguments for trainers
    """

    run_args_trainers = list()
    for i in range(len(private_ip_trainers)):
        if i == 0:
            master_ip = "localhost"
        else:
            master_ip = private_ip_trainers[0]
        info = {"cmd": "rm -rf torchrec_baseline && git clone https://github.com/Jesse-Yan/torchrec_baseline.git && cd torchrec_baseline && bash run_trainers_wdn.sh {} {} {} {}".format(batch_size, master_ip, len(private_ip_trainers), log_file_name)}
        run_args_trainers.append(info)
    return run_args_trainers


def return_args_trainers_dcn(
    batch_size, private_ip_trainers, log_file_name
):
    """
    Arguments for trainers
    """

    run_args_trainers = list()
    for i in range(len(private_ip_trainers)):
        if i == 0:
            master_ip = "localhost"
        else:
            master_ip = private_ip_trainers[0]
        info = {"cmd": "rm -rf torchrec_baseline && git clone https://github.com/Jesse-Yan/torchrec_baseline.git && cd torchrec_baseline && bash run_trainers_dcn.sh {} {} {} {}".format(batch_size, master_ip, len(private_ip_trainers), log_file_name)}
        run_args_trainers.append(info)
    return run_args_trainers


def return_args_trainers_dfm(
    batch_size, private_ip_trainers, log_file_name
):
    """
    Arguments for trainers
    """

    run_args_trainers = list()
    for i in range(len(private_ip_trainers)):
        if i == 0:
            master_ip = "localhost"
        else:
            master_ip = private_ip_trainers[0]
        info = {"cmd": "rm -rf torchrec_baseline && git clone https://github.com/Jesse-Yan/torchrec_baseline.git && cd torchrec_baseline && bash run_trainers_dfm.sh {} {} {} {}".format(batch_size, master_ip, len(private_ip_trainers), log_file_name)}
        run_args_trainers.append(info)
    return run_args_trainers


def return_data_move_args_original(private_ip_trainers):
    run_args_move_files = [
        {
            "cmd": "aws s3 cp s3://recommendation-data-bagpipe/torchrec_dataset ./torchrec_dataset --recursive"
        }
        for i in range(len(private_ip_trainers))
    ]

    return run_args_move_files


def launch_instances_on_demand(launch_cfg):
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])

    instance_lifecycle = launch_cfg["method"]
    instance_count = launch_cfg["instance_count"]

    if instance_lifecycle == "onDemand":
        print("in")
        response = client.run_instances(
            MaxCount=launch_cfg["instance_count"],
            MinCount=launch_cfg["instance_count"],
            ImageId=launch_cfg["ami_id"],
            InstanceType=launch_cfg["instance_type"],
            KeyName=launch_cfg["key_name"],
            EbsOptimized=True,
            IamInstanceProfile={"Name": launch_cfg["iam_role"]},
            # Placement={"AvailabilityZone": launch_cfg["az"]},
            # Placement={"GroupName": launch_cfg["GroupName"]},
            SecurityGroups=launch_cfg["security_group"],
        )
    else:
        print("Not a valid launch method")
        sys.exit()

    instance_ids = list()

    for request in response["Instances"]:
        instance_ids.append(request["InstanceId"])
    time.sleep(5)
    loop = True
    while loop:
        loop = False
        print("Instance ids {}".format(instance_ids))
        response = client.describe_instance_status(
            InstanceIds=instance_ids, IncludeAllInstances=True
        )
        # print("Response {}".format(response))
        for status in response["InstanceStatuses"]:
            print("Status {}".format(status["InstanceState"]["Name"]))
            if status["InstanceState"]["Name"] != "running":
                loop = True
                time.sleep(5)
    print("All instances are running ...")

    instance_collection = ec2.instances.filter(
        Filters=[{"Name": "instance-id", "Values": instance_ids}]
    )
    print("Instance collection {}".format(instance_collection))
    private_ip = []
    public_ip = []
    for instance in instance_collection:
        print(instance.private_ip_address)
        private_ip.append(instance.private_ip_address)
        print(instance.public_ip_address)
        public_ip.append(instance.public_ip_address)
    return (private_ip, public_ip, instance_ids)


def launch_instances_spot(launch_cfg):
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])

    instance_lifecycle = launch_cfg["method"]
    instance_count = launch_cfg["instance_count"]
    launch_dict = {
        "KeyName": launch_cfg["key_name"],
        "ImageId": launch_cfg["ami_id"],
        "InstanceType": launch_cfg["instance_type"],
        "Placement": {"AvailabilityZone": launch_cfg["az"]},
        # "Placement": {"GroupName": launch_cfg["GroupName"]},
        "SecurityGroups": ["pytorch-distributed"],
        "IamInstanceProfile": {"Name": launch_cfg["iam_role"]},
    }

    if instance_lifecycle == "spot":
        response = client.request_spot_instances(
            InstanceCount=launch_cfg["instance_count"],
            LaunchSpecification=launch_dict,
            SpotPrice=launch_cfg["spot_price"],
        )
        print(response)
    else:
        print("Spot is not being used")
        sys.exit()

    request_ids = list()
    for request in response["SpotInstanceRequests"]:
        request_ids.append(request["SpotInstanceRequestId"])

    fulfilled_instances = list()
    loop = True

    print("Waiting for requests to fulfill")
    time.sleep(5)
    while loop:
        request = client.describe_spot_instance_requests(
            SpotInstanceRequestIds=request_ids
        )
        for req in request["SpotInstanceRequests"]:
            print(req)
            if req["State"] in ["closed", "cancelled", "failed"]:
                print("{}:{}".format(req["SpotInstanceRequestId"], req["State"]))
                loop = False
                break
            if "InstanceId" in req and req["InstanceId"]:
                fulfilled_instances.append(req["InstanceId"])
                print(req["InstanceId"] + "running...")
        if len(fulfilled_instances) == launch_cfg["instance_count"]:
            print("All requested instances are fulfilled")
            break
        time.sleep(5)
    if loop == False:
        print("Unable to fulfill all requested instance ..")
        sys.exit()

    while loop:
        loop = False
        response = client.describe_instance_status(InstanceIds=fulfilled_instances)
        for status in response["InstanceStatuses"]:
            if status["InstanceType"]["Name"] != "running":
                loop = True
    print("All instances are running ..")

    # getting host keys

    instance_collection = ec2.instances.filter(
        Filters=[{"Name": "instance-id", "Values": fulfilled_instances}]
    )
    private_ip = []
    public_ip = []
    for instance in instance_collection:
        print(instance.private_ip_address)
        private_ip.append(instance.private_ip_address)
        print(instance.public_ip_address)
        public_ip.append(instance.public_ip_address)
    return (private_ip, public_ip, fulfilled_instances)


def terminate_instances(instance_id, launch_cfg):
    print("Terminating instances ....")
    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])
    instance_collection = ec2.instances.filter(
        Filters=[{"Name": "instance-id", "Values": instance_id}]
    )
    for instance in instance_collection:
        instance.terminate()
    print("Bye Bye instances ...")


def get_az(instance_id, launch_cfg):

    client = boto3.client("ec2", region_name=launch_cfg["region"])
    ec2 = boto3.resource("ec2", region_name=launch_cfg["region"])
    response = client.describe_instance_status(
        InstanceIds=[instance_id], IncludeAllInstances=True
    )

    for status in response["InstanceStatuses"]:
        az_val = status["AvailabilityZone"]
        return az_val


def run_large_scale():

    launch_cfg = {
        "name": "recommendation-setup",
        "key_name": "chengpo_oregon",
        "key_path": "/Users/jesse/Documents/cs-shivaram/chengpo_oregon.pem",
        "region": "us-west-2",
        "method": "onDemand",  # onDemand
        "az": "us-west-2c",
        "GroupName": "distributed-training",
        # "ami_id": "ami-0f07487e2b2761b0a", # nv old
        # "ami_id": "ami-04e4121bc8f056792", # oregon old
        # "ami_id": "ami-00cfdc3a2d9df3424",
        # "ami_id": "ami-0da11783bca01840b",
        "ami_id": "ami-0292fe69108745952",
        "ssh_username": "ubuntu",
        "iam_role": "ec2-s3-final",
        "instance_type": "p3.2xlarge",
        # "instance_type": "t2.medium",
        "instance_count": 2,
        "spot_price": "4.5",
        "security_group": ["pytorch-distributed"],
    }

    num_trainers = 8

    # launching trainers
    launch_cfg["instance_type"] = "g5.8xlarge"
    launch_cfg["method"] = "onDemand"
    launch_cfg["instance_count"] = num_trainers
    (
        private_ip_trainers,
        public_ip_trainers,
        instance_ids_trainers,
    ) = launch_instances_on_demand(launch_cfg)

    p3_az = get_az(instance_ids_trainers[0], launch_cfg)

    # trainer client
    client_trainers = ParallelSSHClient(
        public_ip_trainers, user="ubuntu", pkey=launch_cfg["key_path"]
    )

    # trainer client warmup ebs

    run_args_get_data = return_data_move_args_original(private_ip_trainers)

    time.sleep(60)
    
    output_trainers = client_trainers.run_command(
        "%(cmd)s", host_args=run_args_get_data
    )

    for hosts_out in output_trainers:
        for line in hosts_out.stdout:
            print(line)
    
    global_batch_size = 16384
    batch_size = global_batch_size // num_trainers
    with_dlrm = True
    with_wdn = True
    with_dcn = True
    with_dfm = True
    
    if with_dlrm:
        # ========Launching Bagpipe run 1========================================
        log_file_name = "run_dlrm_{}_num_machines_{}_run_g5".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_dlrm(
            batch_size, private_ip_trainers, log_file_name
        )

        print("Run args trainer {}".format(run_args_trainers))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)
        
    if with_wdn:
        # ========Launching Bagpipe run 1========================================
        log_file_name = "run_wdn_{}_num_machines_{}_run_g5".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_wdn(
            batch_size, private_ip_trainers, log_file_name
        )

        print("Run args trainer {}".format(run_args_trainers))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)
        
    if with_dcn:
        # ========Launching Bagpipe run 1========================================
        log_file_name = "run_dcn_{}_num_machines_{}_run_g5".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_dcn(
            batch_size, private_ip_trainers, log_file_name
        )

        print("Run args trainer {}".format(run_args_trainers))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)
        
    if with_dfm:
        # ========Launching Bagpipe run 1========================================
        log_file_name = "run_dfm_{}_num_machines_{}_run_g5".format(
            len(private_ip_trainers), batch_size
        )
        run_args_trainers = return_args_trainers_dfm(
            batch_size, private_ip_trainers, log_file_name
        )

        print("Run args trainer {}".format(run_args_trainers))

        output_trainers = client_trainers.run_command(
            "%(cmd)s", host_args=run_args_trainers
        )

        for hosts_out in output_trainers:
            for line in hosts_out.stdout:
                print(line)

        time.sleep(60)
    
    terminate_instances(instance_ids_trainers, launch_cfg)

if __name__ == "__main__":
    run_large_scale()
