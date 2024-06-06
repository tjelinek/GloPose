import pydevd_pycharm
pydevd_pycharm.settrace('2001:718:2:1672::208', port=12343, stdoutToServer=True, stderrToServer=True)  # suspend=False


print("Hello World")