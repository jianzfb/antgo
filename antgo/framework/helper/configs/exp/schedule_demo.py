exp=dict(
    launch_script='python3 test_m.py',
    args='',
    schedule=[
        {
            '--checkpoint': '/opt/tiger/handdetJ/checkpoint/greedy_ac_1/handdetJ/epoch_50.pth',
        }
    ]
)