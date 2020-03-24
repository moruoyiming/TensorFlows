import matplotlib.pyplot as plt
import numpy as np

plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
# 横竖坐标
plt.axis([0, 6, 0, 25])
plt.show()

# 0到5之间每隔0.2取一个数
t = np.arange(0., 5., 0.2)
# 红色的破折号，蓝色的方块，绿色的三角形
plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
plt.show()

# text()命令可以被用来在任何位置添加文字，xlabel()、ylabel()、title()被用来在指定位置添加文字。
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\\sigma=15$')
plt.axis([40, 160, 0, 0.03])
# plt.grid[True]
plt.show()

# 注释文本
ax = plt.subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05))
plt.ylim(-2, 2)
plt.show()

# plt.plot(x,y,format_string,**kwargs)
# 说明：
#
# x:x轴数据，列表或数组，可选
# y:y轴数据，列表或数组
# format_string:控制曲线的格式字符串，可选
# **kwargs:第二组或更多，(x,y,format_string)

# 注：当绘制多条曲线时，各条曲线的x不能省略
a = np.arange(10)
plt.plot(a, a * 1.5, a, a * 2.5, a, a * 3.5, a, a * 4.5)
plt.show()
# format_string：控制曲线的格式字符串，可选，由颜色字符、风格字符和标记字符组成。
#
# 颜色字符	说明	    颜色字符	说明
# ‘b’	    蓝色	    ‘m’	    洋红色 magenta
# ‘g’	    绿色	    ‘y’	    黄色
# ‘r’	    红色	    ‘k’	    黑色
# ‘c’	    青绿色    cyan	‘w’	白色
# ‘#008000’	RGB某颜色‘0.8’	灰度值字符串

# 风格字符	说明
# ‘-‘	    实线
# ‘–’	    破折线
# ‘-.’	    点划线
# ‘:’	    虚线
# ’ ’ ’ ‘	无线条

# 标记字符	说明	            标记字符	    说明
# ‘.’	    点标记	        ‘1’	        下花三角标记
# ‘,’	    像素标记（极小点）	‘2’	        上花三角标记
# ‘o’	    实心圈标记	    ‘3’	        左花三角标记
# ‘v’	    倒三角标记	    ‘4’	        右花三角标记
# ‘^’	    上三角标记	    ’s’	        实心方形标记
# ‘>’	    右三角标记	    ‘p’	        实心五角标记
# ‘<’	    左三角标记	    ‘*’	        星形标记
# ‘h’	    竖六边形标记
# ‘H’	    横六边形标记
# ‘+’	    十字标记
# ‘x’	    x标记
# ‘D’	    菱形标记
# ‘d’	    瘦菱形标记
# ‘|’	    垂直线标记

b = np.arange(10)
plt.plot(a, a * 1.5, 'go-', a, a * 2.5, 'rx', a, a * 3.5, '*')
plt.show()

# 注：
# plt.plot(x,y,format_string,**kwargs)
#
# **kwargs:第二组或更多，(x,y,format_string)
# color:控制颜色，color=’green’
# linestyle:线条风格，linestyle=’dashed’
# marker:标记风格，marker = ‘o’
# markerfacecolor:标记颜色，markerfacecolor = ‘blue’
# markersize:标记尺寸，markersize = ‘20’
