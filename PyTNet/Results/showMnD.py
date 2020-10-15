"""

displays the actual and predicted images of both mask and depth


"""
import torch
import matplotlib.pyplot as plt
import torchvision
def show_result_img(target, predict, type, name):
    
    fig, a =plt.subplots(2,1,figsize=(45,35))
    fig.suptitle(type +" "+ name,fontweight="bold",fontsize=45,y=1.1,color='r')
    
 
    target= target*255
    target = target.numpy()

    predict = predict*255
    predict = predict.numpy()

  
    plt.axis("off")
    a[0].imshow(target[0], cmap = "gray")
    a[1].imshow(predict[0], cmap = "gray")
   
    a[0].set_title('Target '+type,fontsize=35)
    a[1].set_title('Predicted ' +type,fontsize=35)

    a[0].axis("off")
    a[1].axis("off")

  
        
      
   
  
    plt.savefig(name+'_'+type+'.jpg')
    plt.tight_layout()
    plt.show()
    print(f"Results are saved in {name}_ {'type'}.jpg")

   
  

def show_results(model,testloader,name):
    batch = next(iter(testloader))
    images,mask_target,depth_target = batch
    mask, depth = model(images.to(device))

    batch_preds_mask = torch.sigmoid(mask)
    batch_preds_mask = batch_preds_mask.detach().cpu()
    
    batch_preds_depth = depth
    batch_preds_depth = batch_preds_depth.detach().cpu()


    plt.axis("off")
    m = torch.unsqueeze(mask_target, 1)
    d = torch.unsqueeze(depth_target, 1)
    images = []
    mask_target = []
    depth_target = []
    mask_pred = []
    depth_pred = []
    for i in range(20):
      mask_target.append(m[i])
      depth_target.append(d[i])
      mask_pred.append(batch_preds_mask[i])
      depth_pred.append(batch_preds_depth[i])
    mask_t = torchvision.utils.make_grid(mask_target,nrow=5,padding=1,scale_each=True)
    mask_p =torchvision.utils.make_grid(mask_pred,nrow=5,padding=1,scale_each=True)
    depth_t = torchvision.utils.make_grid(depth_target,nrow=5,padding=1,scale_each=True)
    depth_p = torchvision.utils.make_grid(depth_pred,nrow=5,padding=1,scale_each=True)
    show_result_img(mask_t,mask_p, "mask", name = name)
    show_result_img(depth_t,depth_p, "depth", name = name)
# show_results(model,testloader,name = "Results_2")

     