import torchgeometry as tgm


class BCESSIM():
  def __init__(self ):
    super(BCESSIM, self).__init__()
    self.ssim = tgm.looses.SSIM(window_size = 3, reduction = mean)

  def forward(self,input,target):
    ssim = self.ssim(input,target)
    bce = F.binary_cross_entropy_with_logits(input, target)
    loss = bce + 2*ssim

    return loss

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss