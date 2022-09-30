
def sync_model(self):

    model_name = None
    self.sync_weights_count += 1
    if self.sync_weights_count >= self.broadcast_weights_interval:
        model_name = self.recv_explorer.recv(block=False)
        self.sync_weights_count = 0

    model_successor = model_name
    while model_successor:
        model_successor = self.recv_explorer.recv(block=False)
        if model_successor is not None:
            model_name = model_successor

    return model_name
