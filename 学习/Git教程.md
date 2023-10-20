# Git教程

## 一.Git

### 1.1 命令

- 初始化`git init`
- 状态查看`git status`
- 暂存区添加`git add 文件名`
- 删除暂存区文件`git rm  --cached 文件名`、
- 提交暂存区到本地库`git commit -m "restudy git"`

```
$ git commit -m "restudy git"
[master (root-commit) 8a88eed] restudy git
 12 files changed, 1426 insertions(+)
 create mode 100644 help.cpp
 create mode 100644 help.h
 create mode 100644 help.ui
 create mode 100644 img/background.png
 create mode 100644 main.cpp
 create mode 100644 snake.pro
 create mode 100644 snake.pro.user
 create mode 100644 snake.pro.user.8ed1e93
 create mode 100644 src.qrc
 create mode 100644 widget.cpp
 create mode 100644 widget.h
 create mode 100644 widget.ui
```

> 100644-->版本码

- 日志查看


`git reflog`：显示commit和reset操作的版本号。可以用来恢复错误回退操作，查询出所有的相关操作。

`git log`：只显示已提交的版本号，不包括已删除的提交和回退操作[git log和git reflog的区别](https://blog.csdn.net/chenpuzhen/article/details/92084229)

- 修改文件


首先查看要修改的文件`cat 文件名`获取文件内容，然后使用vim编辑器（目前还不会用）`vim 文件名`进入vim编辑器详vim教程

版本穿梭 `git reset --hard 1603982`后缀为版本号，可以通过git reflog命令获取版本号

### 1.2 分支操作

- 创建分支`git branch 分支名`

- 查看分支`git branch -v`或者`git branch`

- 切换分支`git checkout 分支名`

- 合并分支`git merge 分支1`:将分支1合并到现分支。合并分支存在两种情况，正常合并和冲突合并。无论是那种合并方式，都使用给

  git merge命令。正常合并时，会弹出形如：

```
Updating 1603982..0615c58
Fast-forward
 两个版本不同的文件名 | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

的语句，表明现如今是正常合并，没有冲突存在。当出现代码冲突时，就会出现形如：自动合并分支时，help.cpp文件出现冲突，这时分支名中还会出现`(分支名|MERGING)`正在合并的标注

```
Auto-merging 冲突文件名
CONFLICT (content): Merge conflict in 冲突文件名
Automatic merge failed; fix conflicts and then commit the result.
```

紧接着查询git状态

```
On branch 合并分支名
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)
        both modified:   冲突文件名

no changes added to commit (use "git add" and/or "git commit -a")
```

然后在vim编辑器中修改代码，git会在冲突代码处添加形如：

```
<<<<<<< HEAD
//在热修分支上修改文件,测试分支冲突时的解决方案
=======
//测试呀
>>>>>>> 被合并分支名
```

的代码块，在手动修改之后合并代码，去除`<<<<<<< HEAD`和`=======`还有`>>>>>>> hot-fix`保证冲突代码区域总行数不变。在手动合并完冲突代码之后，需要在合并分支上重新add和commit到本地库中。至此合并分支名后的MERGING消除，冲突代码手动合并完成。

### 1.3 团队协作

#### 1.3.1 团队内协作

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230713090945448.png" alt="image-20230713090945448" style="zoom:50%;" />

#### 1.3.2 跨团队协作

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230713091947828.png" alt="image-20230713091947828" style="zoom:50%;" />

## 二.Github

### 2.1 创建远程仓库

- 起远程仓库别名：在github上注册账号，建立一个远程仓库。获取https信息，使用`git remote add 别名 https地址`命令建立与远程仓库的链接，然后使用`git remote -v`产看远程仓库信息获取到

```
git-to-c-test   https://github.com/Lubiyangka/test--c.git (fetch)
//别名           HTTPS地址                                  拉取
git-to-c-test   https://github.com/Lubiyangka/test--c.git (push)
//别名           HTTPS地址                                  推送
```

- 推送&拉取：推送操作使用如下两种命令即可`git push 仓库别名 分支名`或`git push HTTPS地址 分支名` ，拉取时同推送操作基本一致，只是将push换成pull，也有两种方法`git pull 仓库别名 分支名`或`git pull HTTPS地址 分支名` 
- 克隆代码：在文件夹中启动git bash然后使用`git clone HTTPS地址`就可以在该文件下的子目录中获取到克隆下的代码，`cd 仓库名`进入仓库名文件夹，`git status`获取仓库状态，`git checkout  分支名`切换到你想要的分支。

## 三.Gitlab

## 四.Gitee