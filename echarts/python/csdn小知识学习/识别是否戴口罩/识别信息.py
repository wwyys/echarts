import paddlehub as hub
#加载模型
module=hub.Module(name='pyramidbox_lite_mobile_mask')
#图片列表
image_list=['face.jpg']
#获取图片字典
input_dict={'image':image_list}
#检测是否带了口罩
module.face_detection(data=input_dict)