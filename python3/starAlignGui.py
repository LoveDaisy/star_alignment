from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.config import Config
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.graphics import Color,Rectangle
import numpy as np
import cv2
import tifffile as tiff

class starAlignBL(BoxLayout):
    img=None
    texture=None
    
    def test_draw(self):
        self.img=cv2.flip(tiff.imread("DSC02690.tif"),0)
##        img=(img.astype("float32") / np.iinfo(img.dtype).max * 256).astype("uint8")
##        img=img.astype("float32") / np.iinfo(img.dtype).max
##        img=cv2.imread("DSC02690.tif")
        self.texture = Texture.create(size=(self.img.shape[1],self.img.shape[0]),
                                 colorfmt='rgb',
                                 bufferfmt='ushort')
        self.texture.blit_buffer(self.img.tobytes(),bufferfmt='ushort')
        render_texture=self.ids['render_area']

        img_ratio=self.img.shape[1]/self.img.shape[0]
        render_ratio=render_texture.size[0]/render_texture.size[1]
        if img_ratio > render_ratio:
            render_size=(render_texture.size[1]*img_ratio,render_texture.size[1])
        else:
            render_size=(render_texture.size[0],render_texture.size[0]/img_ratio)
          
        render_texture.size=render_size
        with render_texture.canvas:
            Rectangle(texture=self.texture,pos=render_texture.pos, size=render_size)

    def show_on_size(self):
        if self.img is not None:
            render_texture=self.ids['render_area']

            img_ratio=self.img.shape[1]/self.img.shape[0]
            render_ratio=render_texture.size[0]/render_texture.size[1]
            if img_ratio > render_ratio:
                render_size=(render_texture.size[1]*img_ratio,render_texture.size[1])
            else:
                render_size=(render_texture.size[0],render_texture.size[0]/img_ratio)
              
            render_texture.size=render_size
            with render_texture.canvas:
                Rectangle(texture=self.texture,pos=render_texture.pos, size=render_size)


            
##    def loadImage(self,fileList):
##        pass
##
##    def debug(self, msg):
##        pass
##
##    def warning(self, msg):
##        pass
##
##    def error(self, msg):
##        '''
##        error(self,msg)
##            A logging function for youtube_dl, will display the error status
##            Argument: msg passed in by youtube-dl
##        '''
##        pass
##
##    #@mainthread
##    def prog_hook(self,d):
##        pass
##
##    def merge(self, *args):
##        pass

class starAlignApp(App):
    def build(self):
        return starAlignBL()


if __name__ == "__main__":
    Config.set("kivy","exit_on_escape","0")
    starAlignApp().run()
