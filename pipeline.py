class ClipY:
    def __call__(self, sample):

        u, v, y, index = sample["u"], sample["v"], sample["y"], sample["index"]

        if y > 3751.3125:
            y = 3751.3125

        y = 1 + np.log(1+y)
        sample = {"u":u, "v":v, "y": torch.tensor(y).float(), "index":index}
        
        return sample
