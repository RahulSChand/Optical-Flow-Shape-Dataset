import cv2
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import math
import random
from utils import show_flow
import os
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

def combine_img_1(img,img2,center,radius,shape='circle',threshold=255):
    
    rot = random.randint(0,40) - 20
    #rot=10
    rot_prev=rot
    #get list of coordinates and give it a random color mapping
    threshold - (255,255,255)
    if shape=='hex':
        
        img = draw_hexagon(img,center,radius,color=threshold,start_angle=rot)
    if shape=='square':
        img = draw_square(img,center,radius,color=threshold,start_angle=rot)
    if shape=='triangle':
        img = draw_triangle(img,center,radius,color=threshold,start_angle=rot)
    if shape=='circle':
        img = draw_circle(img,center,radius,color=threshold,start_angle=0)
    if shape=='star':
        img = draw_star(img,center,radius,color=threshold,start_angle=rot)
    

    rot = random.randint(0,40) - 20
    #rot=15
    translate = [random.randint(20,45),random.randint(20,45)]

    neg1 = random.randint(0,1)
    
    if neg1==1:
        translate[0] = translate[0]*-1
    neg2 = random.randint(0,1)
    
    if neg2==1:
        translate[1] = translate[1]*-1
    
    #print(neg1,neg2)


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

    optical_flow[0] = optical_flow_x - center[0]
    optical_flow[1] = optical_flow_y - center[1]
    
    optical_flow[0] = optical_flow[0]*math.cos(get_radians(rot)) + optical_flow[1]*math.sin(get_radians(rot))
    optical_flow[1] = optical_flow[1]*math.cos(get_radians(rot)) - optical_flow[0]*math.sin(get_radians(rot))

    optical_flow[0] = optical_flow[0] + center[0]
    optical_flow[1] = optical_flow[1] + center[1]

    optical_flow[0] = optical_flow[0] + translate[0]
    optical_flow[1] = optical_flow[1] + translate[1]
    
    optical_flow[0] = optical_flow[0] - optical_flow_x
    optical_flow[1] = optical_flow[1] - optical_flow_y

    #(H,W,3) or (3,H,W)?
    boolMap = np.sum(np.array(img),axis=2)<255*3
    colorMap = np.random.randint(0,255,size=(128,128,3))
    colorMap[boolMap] = 0

    optical_flow[0][boolMap] = 0
    optical_flow[1][boolMap] = 0

    center[0] = center[0] + translate[0]
    center[1] = center[1] + translate[1]
    
    if shape=='hex':
        img2 = draw_hexagon(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='square':
        img2 = draw_square(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='triangle':
        img2 = draw_triangle(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='circle':
        img2 = draw_circle(img2,center,radius,color=threshold,start_angle=rot+rot_prev)
    if shape=='star':
        img2 = draw_star(img2,center,radius,color=threshold,start_angle=rot+rot_prev)

    img2 = x

    return img,img2,optical_flow

def combine_img_2(img,center,radius):

    img1 = Image.new('L',(128,128))
    radius = radius*0.8
    r = random.randint(5,15)
    ##print(r)
    img1 = draw_circle(img1,center,radius,color=255)
    img1 = draw_circle(img1,center,max(radius-5,5),color=0)
    img.paste(img1)
    return img


'''
s=0
for i in range(s,s+128):
    
    img_whole = Image.new('L',(128,128))
    img_whole2 = Image.new('L',(128,128))
    c1 = random.randint(25,100)
    c2 = random.randint(15,113)
    r = random.randint(14,40)
    
    img_whole,img_whole2,op_flow_1 = combine_img_1(img_whole,img_whole2,[c1,c2],r,'hex',threshold=254)

    c1=100-c1
    c2=113-c2
    
    img_whole,img_whole2,op_flow_2 = combine_img_1(img_whole,img_whole2,[c1,c2],r,'star',threshold=255)

    img_whole.save('shapes_double_recon/'+str(i)+'.png','PNG')
'''

'''
s=700
for i in range(s,s+100):
    
    img_whole = Image.new('L',(128,128))
    img_whole2 = Image.new('L',(128,128))
    c1 = random.randint(25,100)
    c2 = random.randint(15,113)
    r = random.randint(14,40)

    img_whole,img_whole2,op_flow_1 = combine_img_1(img_whole,img_whole2,[c1,c2],r,'circle',threshold=254)

    c1=100-c1
    c2=113-c2
    img_whole,img_whole2,op_flow_2 = combine_img_1(img_whole,img_whole2,[c1,c2],r,'star',threshold=255)

    op_flow = op_flow_2+op_flow_1

    img_whole.save('shapes_double2/'+str(i)+'_1.png','PNG')
    img_whole2.save('shapes_double2/'+str(i)+'_2.png','PNG')  

    img_flow = show_flow(op_flow)
    cv2.imwrite('shapes_double2/'+str(i)+'_flow.png',img_flow)

    np.save('shapes_double2/'+str(i)+'_flow.npy',op_flow)
'''

'''
#Create dataset to test on shapes model
name = ['hex','square','triangle','circle','star']
for k in range(5):
    s=0
    for i in range(s,s+250):
        img_whole = Image.new('L',(128,128))
        img_whole2 = Image.new('L',(128,128))
        c1 = random.randint(25,100)
        c2 = random.randint(15,113)
        r = random.randint(14,40)
        
        img_whole,img_whole2,op_flow = combine_img_1(img_whole,img_whole2,[c1,c2],r,name[k])
        path = 'shapes_test_fast/'+name[k]
        os.makedirs(path, exist_ok=True)
        img_whole.save(path+'/'+str(i)+'_1.png','PNG')
        img_whole2.save(path+'/'+str(i)+'_2.png','PNG')  
        
        img_flow = show_flow(op_flow)

        cv2.imwrite(path+'/'+str(i)+'_flow.png',img_flow)
        np.save(path+'/'+str(i)+'_flow.npy',op_flow)

'''