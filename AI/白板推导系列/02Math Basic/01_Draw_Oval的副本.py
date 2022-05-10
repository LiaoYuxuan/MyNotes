import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.axisartist as ast
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/Library/Fonts/Songti.ttc")

# 长短轴
a, b = (2. ,2. )
# 需要使用参数方程
t = np.arange(0, 2*np.pi, 0.01)
x = a*np.cos(t)
y = b*np.sin(t)
x_t = (a+1)*np.cos(t)
y_t = (b+1)*np.sin(t)

# 在图中绘制坐标系
fig = plt.figure(figsize=(10,10))
ax = ast.Subplot(fig, 1,1,1)
fig.add_axes(ax)
# 隐藏plt自带的坐标轴
ax.axis[:].set_visible(False)
# 设置新坐标轴的位置和参数
ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["x"].set_axisline_style("->", size=1.5)
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["y"].set_axisline_style("->", size=1.5)
ax.axis["x"].set_axis_direction("top")
ax.axis["y"].set_axis_direction("right")
# 设置坐标轴名字
ax.text(5.5, 0.15, '$x_1$', fontsize=10)
ax.text(0.1, 5.5, '$x_2$', fontsize=10)
ax.text(0.5, 1.4, '$\sqrt{\lambda_1}$', fontsize=20, color="r")
ax.text(2, 3, '$\sqrt{\lambda_2}$', fontsize=20, color="g")
ax.text(3.5, 3.5, '$\Delta_1$', fontsize=20, color="steelblue")
ax.text(4.2, 4.2, '$\Delta_2$', fontsize=20, color="tomato")
# ax.axis["y"].label.set_axis_direction("left")


# 通过矩阵运算实现旋转和平移
# rotAngle = 1/4*np.pi
rotAngle = 0*np.pi
xx = np.cos(rotAngle)*x - np.sin(rotAngle)*y + 2
yy = np.sin(rotAngle)*x + np.cos(rotAngle)*y + 2
xx_t = np.cos(rotAngle)*x_t - np.sin(rotAngle)*y_t + 2
yy_t = np.sin(rotAngle)*x_t + np.cos(rotAngle)*y_t + 2

x_1 = np.array([-a*np.cos(rotAngle), 0]) + 2
y_1 = np.array([-a*np.sin(rotAngle), 0]) + 2
x_2 = np.array([-b*np.sin(rotAngle), 0]) + 2
y_2 = np.array([b*np.cos(rotAngle), 0]) + 2

# 绘制椭圆形
# plt.plot(xx, yy)
plt.plot(xx, yy, color="steelblue")
plt.plot(xx_t, yy_t, dashes=[6, 2], color="tomato")
plt.plot(x_1, y_1, color='r')
plt.plot(x_2, y_2, color='g')
plt.title("各向同性示意图", fontproperties=font, fontsize=15)
plt.show()
# fig.savefig("Gaussian Figure.png")
fig.savefig("Gaussian Figure 3.png")
