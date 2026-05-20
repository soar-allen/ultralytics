# CVAT 托盘标注与审核操作指南
你好！欢迎参与托盘图像的标注工作。  
本次工作的主要任务**不是从头开始画框**，系统已经预先跑出了半自动的标签。你的核心工作是**校验并微调**：检查机器标得对不对，把不准的框调准，把漏掉的补上，把多余的删掉。

## 1. 登录系统
1. 连接公司wifi （COTEK-2.4G-Office 或者 COTEK-5G-Office）
2. 打开浏览器（建议使用 **Google Chrome或者Microsoft Edge**）。
3. 在地址栏输入标注系统的网址： `[http://192.168.1.89:8080/](http://192.168.1.89:8080/)`并回车。
4. 输入分配给你的**账号**和**密码**，点击 Login 登录。
    - **账号**：可用账号`test, test1, test2, test3, test4, test5`
    - **密码**：Cotek@123

## 2. 找到你的任务
1. 登录后，右上角点击用户下拉框，点击`Organizaion`选择 `cotek`(如果已经是`cotek`则不用管)

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1773280652426-cfb3e7be-1d1c-4024-88f9-95e77d4704c3.png" width="351.6" title="" crop="0,0,1,1" id="ue839e127" class="ne-image">

2. 上方的导航栏，点击 **Jobs**（作业）。
3. 找到**Assignee**为自己账号的作业，点击图片即可进入标注界面。  
 <img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1773194869604-0f519bd7-472b-4240-9caa-01f58908bab3.png" width="1920" title="null" crop="0,0,1,1" id="PGRtt" class="ne-image">
4. 如果 `job`被标记为已完成，则不用标注

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1774258719214-6175ea7a-22a4-4070-8c93-9740a1e9c778.png?x-oss-process=image%2Fcrop%2Cx_362%2Cy_0%2Cw_2188%2Ch_1243" width="1750" title="" crop="0.142,0,1,1" id="uf4ebe65c" class="ne-image">

## 3. 工作界面快速认识
进入 Job 后，你会看到主工作区。

+ **中间大图**：你要检查的图片。
+ **左侧工具栏**：用来新建标签（如果发现漏标的话）。
+ **右侧面板**：显示当前图片里已经存在的标签列表（`pallet`  等等）。
+ **顶部导航**：有保存按钮，以及切换上一张/下一张图片的箭头。  
<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1773194869805-10323435-04d8-4bc8-b2c0-9286b0cf0f8b.png" width="1920" title="null" crop="0,0,1,1" id="sReEK" class="ne-image">

## 4. 你的核心工作：审核与微调
系统已经预先框出了大部分托盘，你需要对每一张图片进行以下三步检查：

### 第一步：检查“准不准”（微调边界）
+ **标准**：框的边缘必须**紧紧贴合**托盘的物理边缘，不能切掉托盘的角，也不能包含太多的背景留白。
+ **怎么做**：把鼠标放在多边形框的角点，按住左键拖动，调整多边形的角点贴合托盘的前表面。  
<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1773194870122-e967211b-a35d-4e2b-9097-dc8b8a6cfbea.png" width="1920" title="null" crop="0,0,1,1" id="Ke11G" class="ne-image">

### 第二步：检查“多没多”（删除误报）
+ **标准**：系统有时会把地上的纸箱、木板、货架或阴影错认成托盘。
+ **怎么做**：鼠标点击那个错误的框（或者在右侧列表中找到它），按键盘上的 `Delete` 键删除。

### 第三步：检查“漏没漏”（补充漏报）
+ **标准**：画面中明明有托盘，但是系统没有框出来。
+ **怎么做**：
    1. 在左侧工具栏点击 **Draw new polygon**（画新多边形）。
    2. 选择正确的托盘标签 `川字托盘`（<font style="color:#DF2A3F;">chuan</font>）或者 `田字托盘`（<font style="color:#DF2A3F;">tian</font>）。（图中川字托盘的侧边也可以认为田字托盘）

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775045872856-8b6ec5a1-90ee-4a65-806d-aa27d3f48c34.png" width="1240.8" title="" crop="0,0,1,1" id="u7a36c030" class="ne-image">

    3. 输入 **Number of points** 为 **4**
    4. 点 `shape` 既可以在图片上画多边形
    5. 按照`左上->右上->右下->左下`的顺序用鼠标在漏掉的托盘四角分别点一下，把它严丝合缝地框起来。

**提示**：切换到下一张图片之后，**<font style="color:#DF2A3F;">直接按键盘 </font>**`**<font style="color:#DF2A3F;">N</font>**` 既可以开始标注 (类别为默认为上次操作的类别，如果需要切换类别则需要左侧重新选择)（首次打开作业需要按照上面步骤操作之后才能画多边形）

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775033575096-d30b87e6-d706-4b85-bd0f-ce3a5172eec9.png" width="1065.6" title="" crop="0,0,1,1" id="ud3700510" class="ne-image">

### 常见场景（<font style="color:#DF2A3F;">注意事项</font>）
**核心思想**：侧面可见部分看不到**<font style="color:#DF2A3F;">托盘一面的叉取特征</font>**(对托盘叉取面来说，特征为托盘的两个叉孔以及三个腿)时可以不标注

+ 托盘被<font style="color:#DF2A3F;">严重遮挡</font>，看不清结构时

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775044293261-91c5b959-9549-4309-b8cf-76777392295f.png" width="632.8" title="" crop="0,0,1,1" id="udaca0328" class="ne-image">

+ **托盘****<font style="color:#000000;">明显可见但是被轻微遮挡</font>**<font style="color:#000000;"> </font>，<font style="color:#DF2A3F;">需要标注</font>  
<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775044715676-e98b9610-b17f-4146-b902-d04414b882b9.png" width="323.2" title="" crop="0,0,1,1" id="u517a9a00" class="ne-image">
    - 被遮挡的角点标记为遮挡——选中框，按下鼠标右键，点击 `DETAILS`下拉框勾选被遮挡的点
    - 托盘入叉面按照顺时针的顺序记为 1（左上），2（右上），3（右下），4（左下），如下图
    - <img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775550760713-be8b6ffe-6e8c-473d-ab03-d90fb5fd7442.png" width="968.8" title="" crop="0,0,1,1" id="uc97c227e" class="ne-image">
    - 上图中3号角点被遮挡，则勾选如下图
    - <img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775550850711-c0e4b29c-14d8-41e1-a568-906df5041a27.png" width="956" title="" crop="0,0,1,1" id="u183b9d16" class="ne-image">





+ **托盘位于图像边界，入叉面视角不完整时**（放大图片并移动到屏幕中心，**<font style="color:#DF2A3F;">托盘不完整一侧的两个角点需要超出图像</font>**，框会自动收缩到图像内，如下图）

<img src="https://cdn.nlark.com/yuque/0/2026/gif/34511665/1774328800126-75e34b2a-4745-4ab4-a9f9-bc9980526f3b.gif" width="568" title="" crop="0,0,1,1" id="u24810fca" class="ne-image">

+ **托盘位于图像边界，**但是可见部分<font style="color:#DF2A3F;">过小，可以不标注</font>（例如下图最右侧，可见部分只有一个腿，一个叉孔都不完整），如果<font style="color:#DF2A3F;">托盘可见部分超过一个插孔</font>，则需要标注

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1774329484737-5602dc1f-292f-47f9-93b7-2a26541bce9d.png" width="662" title="" crop="0,0,1,1" id="u3e45a95a" class="ne-image">  


+ 侧面视角可能从非常小的一部分逐渐变大，**<font style="color:#DF2A3F;">当侧面框的大小接近正面框的（1/5）时</font>**，开始标法注侧面（**这个时候需要标注托盘的****<font style="color:#DF2A3F;">两面，根据实际托盘情况选择对应的类别</font>**）

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775037677755-c2bb9b2d-7873-4d18-97f8-e481667c19b2.png" width="953.6" title="" crop="0,0,1,1" id="pNMwq" class="ne-image">

+ 托盘的顶点标注需要准确且一致，一个托盘的两面框不要留缝隙，下图为**<font style="color:#DF2A3F;">错误示范</font>**

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1775028237705-d9da9be1-8974-41ce-a12a-41ab79118e1b.png" width="889.6" title="" crop="0,0,1,1" id="udc5e1a79" class="ne-image">

+ 上下不齐的托盘依旧按照矩形标注，如下图

<img src="https://cdn.nlark.com/yuque/0/2026/png/34511665/1774330507262-87fa3f42-2251-4b98-90f6-4b560c902d5d.png" width="915.3333333333334" title="" crop="0,0,1,1" id="u723fa2ac" class="ne-image">

+ 

## 5. 提升效率的快捷键（必背！）
熟练使用这几个按键，你的速度会提升三倍以上：

+ `F`：下一张图片 (Next frame)
+ `D`：上一张图片 (Previous frame)
+ `Ctrl + S`：保存 (Save) —— **<font style="color:#DF2A3F;">建议每做完几张图就顺手按一下，防止网页崩溃白干！</font>**
+ `Delete`** / **`Backspace`：删除当前选中的框
+ `N`：开始画框标注（取决于上一次侧边栏选择的什么形状）
+ **鼠标滚轮**：放大/缩小图片（遇到看不清的细节，**<font style="color:#DF2A3F;">一定要放大看</font>**！）
+ **按住鼠标左键拖动图片**：在放大的图片上移动视野
+ **鼠标左键双击图片**：恢复图片为默认大小

## 6. 保存与提交
1. 当你完成了一个 Job 里所有的图片后，**<font style="color:#DF2A3F;">再次按下 </font>**`**<font style="color:#DF2A3F;">Ctrl + S</font>**`**<font style="color:#DF2A3F;"> 确保完全保存</font>**。
2. 点击左上角的 **Menu**（菜单，三条横线图标）。
3. 找到 **Change job state**（更改作业状态），将其从 `New` 或 `In Progress` 修改为 `Completed`（已完成）。
4. 通知你的对接人，这个 Job 已经审核完毕！

---

> **⚠️**** 注意事项**  
如果你在操作中遇到任何不确定的图片（比如被货物严重遮挡的托盘不知道该不该标），请**截图发在工作群里询问**，不要凭感觉瞎猜！这对后续 AGV 视觉识别的准确度非常重要。辛苦大家！
>
