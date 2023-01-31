from antgo.ant import environment

def test_ee(check_path):
    result = environment.hdfs_client.ls(check_path)
    print(result)
    