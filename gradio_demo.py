import os
import gradio as gr
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from model.MVANet import inf_MVANet

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1,1.25], interpolation='bilinear', align_corners=False),
    ]
)

# model dir
ckpt_path = os.path.join(os.getcwd(), "saved_model", "MVANet")


net = inf_MVANet().cuda()

# load pretrained dict from model dir
pretrained_dict = torch.load(os.path.join(ckpt_path, 'Model_80.pth'), map_location='cuda')

model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
net.eval()

def predict(image):
    
    with torch.no_grad():
        
      img = image.convert('RGB')
      w_,h_ = img.size
      img_resize = img.resize([1024,1024],Image.BILINEAR)  
      img_var = Variable(img_transform(img_resize).unsqueeze(0)).cuda()
      mask = []
      for transformer in transforms:  
          rgb_trans = transformer.augment_image(img_var)
          model_output = net(rgb_trans)
          deaug_mask = transformer.deaugment_mask(model_output)
          mask.append(deaug_mask)

      prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
      prediction = prediction.sigmoid()
      prediction = to_pil(prediction.data.squeeze(0).cpu())
      prediction = prediction.resize((w_, h_), Image.BILINEAR)

      img.putalpha(prediction)
      return img

with gr.Blocks(title="Remove background") as app:
    color_state = gr.State(value=False)
    matting_state = gr.State(value=(0, 0, 0))

    gr.HTML("<center><h1>MVANet Remove Background</h1></center>")
    with gr.Row(equal_height=False):
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input image")
            run_btn = gr.Button(value="Remove background", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="pil", label="Image Result")

    run_btn.click(predict, inputs=[input_img],
                  outputs=[output_img])


app.launch(share=True, debug=True, show_error=True, server_port=8888,server_name='0.0.0.0') 
