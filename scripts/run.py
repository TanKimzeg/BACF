import subprocess
import sys

def run_main(py_file: str):
    while True:
        try:
            # 执行 main.py
            result = subprocess.run([sys.executable, py_file], check=True, stdout='./data/output.log')
            # 如果正常退出，结束循环
            if result.returncode == 0:
                print("main.py 正常退出")
                break
        except subprocess.CalledProcessError as e:
            print(f"main.py 运行失败，返回码: {e.returncode}，重新尝试...")

if __name__ == "__main__":
    run_main('main.py')