---
layout: post
tags: [matlab,ee]
published : true
---
https://wenku.baidu.com/view/ea3ad3c408a1284ac85043f5.html    
![](http://lh4.googleusercontent.com/mSlXMJncm_Rkqy4ZWAAmRWiIz9Y7oPB1VA6LCw3z8dCIJsFLmapQkJOrMbRpn1oDW0n_WIpbrLbhLT0h1eXCe_gd6DPapoM0kxRCjBPDhsEHIi-DXaXVQ0jwmqOTiJm7qLIShvev)

这里红圈部分连错了，一根连T，一根连V_dc
https://wenku.baidu.com/view/21593027af45b307e8719732.html

https://wenku.baidu.com/view/42581ff251e79b896802269c.html

http://www.docin.com/p-1001308409-f4.html

![](http://lh4.googleusercontent.com/K3GmyMDA6ZVLF7yetgYndA0FvNggz_hwyv7paiHzXThJe65vKv9-CdPI8TmW7IAhQwCPe0PL-UtBmkvJ5zCCwOcyyO4gdtDPKg2hw7L2yzRKTS2RoTjDHjo-I4uYNjqLfNkbgZNR)




![](http://lh5.googleusercontent.com/blZfw933slZgUyGCAV9w0A8RT0h4PXLW85CPQ6JNJ-ItAEBRsb9PROcudvvdaTRQmwse_-_EBQlg9JUQF51klPUD2tJj4kF-39SnL9JlsX4ncWCy7X6vPjmA1O5pmbYQao_E9MNM)


以上文档可能细节处有小问题.
总结
某种意义上,也可以理解为SPWM和SVPWM是一回事,SPWM是正弦波然后做面积等效,SVPWM基波+三次谐波做面积等效（共同之处在于：交点就是开通关断点），因此利用率高，波形如下：

![](http://lh3.googleusercontent.com/iFn4uvaGvG_3_DHSOpKlzMHD89itI1-Pg12yXDu2XZUBW7DUJwRkMz2jzfI2WWIwA0ZS4WPsBQZ7jY2O_BCQWSzCz0CcA5j6LFgXBxUedKHFOztrzYR_KA4Mb2rbXUl-LMkorYgV)



![](http://lh6.googleusercontent.com/2Hsr66OWS6X2bpkq53ds0e7CorKiGn0MuKNd9asppL5r7jUqgun12gWE6UPgF8gS3NYKZ4clFA6CQr4GL_MjX7E4svikZJkUWPeZqguaPD_91GCgPvlepGPxezYk-T4Ai-Op87Fe)


SPWM:正弦波为调制波,等腰三角形为载波.两者的大小比例(可能是交点的纵坐标/三角形的高)决定了最后输出波形的幅值,正弦波的频率决定了输出的频率.载波频率最好是调制波的20倍以上.
调制比是调制波幅值和载波幅值之比，同时调制所得到的基波分量的幅值也与调制比成正比，在双极性SPWM下，得到基波分量的幅值等于调制比M与输入直流电的二分之一的乘积。因此调制比是一个很重要的概念，它决定了所得到正弦基波的幅值。
SVPWM:仿真中,每隔Ts就判断一次处于哪个扇区,并决定用哪几个矢量合成,也就是说有可能Us转一圈,要经过100个Ts,要判断100次.
三角波的幅值和周期非常重要，周期应该等于Ts，幅值等于Ts的一半。
对于SVPWM来说就不是调制比的概念，载波参数应保持不变，想改变频率幅值只能改变调制波（仿真中的三相输入）。


svpwm中的的调制系数定义： 调制度就是载波频率/调制频率，与精度有关。SPWM里调制波=正弦波，载波=三角波，需要调制的信号与固定幅值和载波频率相交，得出来的就是调试后的波形，面积等效于需要调试的信号。 如果调制比太小，调制后的信号很容易失真，
 
电压利用率：
这是因为直流来源于三相交流整流，以三相线电压380V输入为例，整流后直流为540V。如果100%利用率的话，输出三相线电压应当还是达到三相380V（峰值540V）。而SPWM调制时，以直流中性点为参考，相电压的峰值为直流电压的1/2，即0.5Udc，转换成线电压峰值为sqrt(3)\*Udc/2=0.866Udc，因此三相利用率为0.866
在做spwm逆变器仿真时，得出的基波电压u=0.866ud，符合理论计算。svpwm按理论计算，在调制度为一时，基波电压幅值u=ud.




直流电压利用率一般是逆变器能输出的最大三相交流线电压的基波分量有效值与基值的比值，基值的选取有两种，一种是以直流母线电压作为基值，另一种是以逆变器在不同调制方式能输出的最大基波值作为基值，接触的第一种选取得比较多。提高直流利用率相当于扩大了变频器的容量，但以增大有害谐波分量为代价，所以一般在工程上会进行平衡考虑。


SVPWM和spwm的比较，可以看这篇文献
https://www.researchgate.net/publication/267801173_Simulation_and_comparison_of_SPWM_and_SVPWM_control_for_three_phase_inverter







28335 CCS: 出现进不了中断的问题，检查了寄存器之后发现没问题，重启dsp的cpu问题就解决了。准确的说，重新载入out文件后，dsp有一定几率无法进入中断。重启dsp后问题即解决。
dsp锁相环初始化：https://wenku.baidu.com/view/34dc253276eeaeaad0f33006.html





https://gitee.com/kifish/codes/utvjsc1dalr7ewi53kyn874
