from tensorboard import program

tracking_address = "./logs"  # the path of your log file.
listening_port = "6006"

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address, "--port", listening_port])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
