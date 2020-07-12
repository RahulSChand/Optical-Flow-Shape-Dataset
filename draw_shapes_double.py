import cv2
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import math
import random
from utils import show_flow

PI = math.pi

def get_radians(degree):

    return (degree*PI)/180

def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgridclone = vgrid.clone()
        vgridclone[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgridclone[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        del vgrid

        vgridclone = vgridclone.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgridclone)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgridclone)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask


def get_points(center,radius,ver_num,theta,start_angle=0):
    points=[]
    degree = start_angle
    for i in range(ver_num):
        x = center[0] + math.cos(get_radians(degree))*radius
        y = center[1] + math.sin(get_radians(degree))*radius
        points.append((x,y))
        degree +=theta

    return tuple(points)

def draw_hexagon(img,center,radius,color=255,start_angle=0):

    draw = ImageDraw.Draw(img)
    points = get_points(center,radius,6,60,start_angle)
    ##print(points)
    draw.polygon((points),fill=color)
    return img

def draw_triangle(img,center,radius,color=255,start_angle=0):

    draw = ImageDraw.Draw(img)
    points = get_points(center,radius,3,120,start_angle)
    #print(points)
    draw.polygon((points),fill=color)
    return img


def draw_circle(img,center,radius,color=255,start_angle=0):

    draw = ImageDraw.Draw(img)

    p1 = (center[0]+radius,center[1]+radius)
    p2 = (center[0]-radius,center[1]-radius)
    points = [p2,p1]
    #print(points)
    draw.ellipse(points,fill=color)
    return img


def draw_square(img,center,radius,color=255,start_angle=0):
    draw = ImageDraw.Draw(img)
    p1 = (center[0]+radius,center[1]+radius)
    p2 = (center[0]-radius,center[1]-radius)
    points = [p2,p1]

    draw.rectangle(points,fill=color)
    return img

def draw_star(img,center,radius,color=255,start_angle=0):

    draw = ImageDraw.Draw(img)
    points = get_points(center,radius,3,120,start_angle)
    draw.polygon((points),fill=color)

    points = get_points(center,-radius,3,120,start_angle)
    draw.polygon((points),fill=color)

    return img

def draw_cross(img,center,radius1,radius2,color=255,start_angle=0):

    draw = ImageDraw.Draw(img)

    p1 = (center[0]+radius1,center[1]+radius2)
    p2 = (center[0]-radius1,center[1]-radius2)
    points = [p2,p1]

    draw.rectangle(points,fill=color)
    p1 = (center[0]+radius2,center[1]+radius1)
    p2 = (center[0]-radius2,center[1]-radius1)

    points = [p2,p1]
    draw.rectangle(points,fill=color)
    return img

def combine_img_1(img,img2,center,radius,shape='circle',threshold=255,temp1=None,temp2=None,temp_required=False):

    rot = random.randint(0,40)
    rot_prev=rot
    if shape=='hex':
        img = draw_hexagon(img,center,radius,color=threshold,start_angle=rot)
        if temp_required:
            img_temp_1 = draw_hexagon(temp1,center,radius,color=threshold,start_angle=rot)
    if shape=='square':
        img = draw_square(img,center,radius,color=threshold,start_angle=rot)
        if temp_required:
            img_temp_1 = draw_square(temp1,center,radius,color=threshold,start_angle=rot)
    if shape=='triangle':
        img = draw_triangle(img,center,radius,color=threshold,start_angle=rot)
        if temp_required:
            img_temp_1 = draw_triangle(temp1,center,radius,color=threshold,start_angle=rot)
    if shape=='circle':
        img = draw_circle(img,center,radius,color=threshold,start_angle=0)
        if temp_required:
            img_temp_1 = draw_circle(temp1,center,radius,color=threshold,start_angle=0)
    if shape=='star':
        img = draw_star(img,center,radius,color=threshold,start_angle=rot)
        if temp_required:
            img_temp_1 = draw_star(temp1,center,radius,color=threshold,start_angle=rot)

    rot = random.randint(0,40)
    translate = [random.randint(2,22),random.randint(2,22)]

    neg = random.randint(0,1)
    if neg==1:
        translate[0] = translate[0]*-1
    neg = random.randint(0,1)
    if neg==1:
        translate[1] = translate[1]*-1


    optical_flow = np.zeros([2,128,128])

    optical_flow_x = np.arange(128*128)
    optical_flow_y = np.arange(128*128)

    optical_flow_x = optical_flow_x%128
    optical_flow_y = optical_flow_y%128

    ##print(optical_flow_x)
    ##print(optical_flow_y)


    optical_flow_x = np.reshape(optical_flow_x,[128,128])
    optical_flow_y = np.reshape(optical_flow_y,[128,128])
    optical_flow_y = np.transpose(optical_flow_y,[1,0])

    ##print(optical_flow_x)
    ##print(optical_flow_y)
    if shape=='circle':
        rot=0
        rot_prev=0

    optical_flow[0] = optical_flow_x*(math.cos(get_radians(rot))-1) - optical_flow_y*(math.sin(get_radians(rot))) + translate[0]
    optical_flow[1] = optical_flow_x*(math.sin(get_radians(rot))) + optical_flow_y*(math.cos(get_radians(rot))-1) + translate[1]

    optical_flow[0][np.array(img)<threshold] = 0
    optical_flow[1][np.array(img)<threshold] = 0

    center[0] = center[0] + translate[0]
    center[1] = center[1] + translate[1]

    if shape=='hex':
        img2 = draw_hexagon(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
        if temp_required:
            img_temp_2 = draw_hexagon(temp2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='square':
        img2 = draw_square(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
        if temp_required:
            img_temp_2 = draw_square(temp2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='triangle':
        img2 = draw_triangle(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
        if temp_required:
            img_temp_2 = draw_triangle(temp2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='circle':
        img2 = draw_circle(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
        if temp_required:
            img_temp_2 = draw_circle(temp2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='star':
        img2 = draw_star(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
        if temp_required:
            img_temp_2 = draw_star(temp2,center,radius,color=threshold,start_angle=rot+rot_prev)

    if temp_required==False:
        img_temp_1,img_temp_2 = None,None

    return img,img2,optical_flow,img_temp_1,img_temp_2

def combine_img_2(img,center,radius):

    img1 = Image.new('L',(128,128))
    radius = radius*0.8
    r = random.randint(5,15)
    ##print(r)
    img1 = draw_circle(img1,center,radius,color=255)
    img1 = draw_circle(img1,center,max(radius-5,5),color=0)
    img.paste(img1)
    return img

def get_double_image(shape1,shape2):

    img_whole = Image.new('L',(128,128))
    img_whole2 = Image.new('L',(128,128))

    img_whole_temp = Image.new('L',(128,128))
    img_whole2_temp = Image.new('L',(128,128))


    c1 = random.randint(25,100)
    c2 = random.randint(15,113)
    r = random.randint(14,40)

    img_whole,img_whole2,op_flow_1,_,_ = combine_img_1(img_whole,img_whole2,[c1,c2],r,shape1,threshold=254,temp1=None,temp2=None,temp_required=False)

    c1 = random.randint(25,100)
    c2 = random.randint(15,113)
    
    #c1=100-c1
    #c2=113-c2

    img_first_1 = img_whole.copy()
    r = random.randint(14,40)
    #img_first_2 = img_whole2.copy()

    img_whole,img_whole2,op_flow_2,temp1,temp2 = combine_img_1(img_whole,img_whole2,[c1,c2],r,shape2,threshold=255,temp1=img_whole_temp,temp2=img_whole2_temp,temp_required=True)
    
    #op_flow_1 --> flow for img1?
    #op_flow_2 --> flow for img2?
    #Yes
    
    flow_1_boolean = op_flow_1!=0.0
    flow_2_boolean = op_flow_2!=0.0
    flow_boolean = flow_1_boolean*flow_2_boolean
    flow_boolean = 1 - flow_boolean
    op_flow_2 = op_flow_2*flow_boolean
    op_flow_1 = op_flow_1 + op_flow_2

    return img_whole,img_whole2,op_flow_1,op_flow_2,img_first_1,temp1


s=50
for i in range(s,s+50):
    img_whole = Image.new('L',(128,128))
    img_whole2 = Image.new('L',(128,128))
    c1 = random.randint(25,100)
    c2 = random.randint(15,113)
    r = random.randint(14,40)

    img_whole,img_whole2,flow1,flow2,_,_ = get_double_image('circle','triangle')

    img_whole.save('shapes_double_t/'+str(i)+'_1.png','PNG')
    img_whole2.save('shapes_double_t/'+str(i)+'_2.png','PNG')

    flow_1_boolean = flow1!=0.0
    flow_2_boolean = flow2!=0.0
    flow_boolean = flow_1_boolean*flow_2_boolean
    flow_boolean = 1 - flow_boolean
    flow2 = flow2*flow_boolean
    flow1 = flow1 + flow2

    img_flow1 = show_flow(flow1)
    #img_flow2 = show_flow(flow2)

    cv2.imwrite('shapes_double_t/'+str(i)+'_flow1.png',img_flow1)
    #cv2.imwrite('shapes_double_t/'+str(i)+'_flow2.png',img_flow2)

    #np.save('shapes_double/'+str(i)+'_flow.npy',op_flow)

