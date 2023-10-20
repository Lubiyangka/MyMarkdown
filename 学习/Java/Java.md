[[Spring]][[持久层框架]]
# 一 面向对象
## 1.1 多态性
`instanceof`运算符：判断类变量指向的具体实例，既可以是本类的实例，也可以是子类的实例。
```Java
public class Employee{
	public void fun(Employee e){
		if(e instanceof Manager){
			//子类1，相关操作
		}else if(e instanceof Contractor){
			//子类2，相关操作
		}else{
			//普通雇员
		}
	}
}
```
在子类操作中可以将父类对象，转换成子类对象，进行后续操作。
## 1.2 抽象性
### 1.2.1`abstract`
可以通过关键字`abstract`把一个类定义为抽象类。在抽象类中，每一个未被定义具体实现的方法也标记 为abstract，称为 __抽象方法__. 在程序中不能用抽象类作为模板来创建对象，必须生 成抽象类的一个非抽象的子类后才能创建实例。抽象类可以包含抽象方法 和非抽象方法 ，反之，不能 在非抽象类中声明抽象方法，即只有抽象类才能具有 抽象方法。
```Java 
public abstract class O{
	int om = 0;
	Object s[] = new Object[100];
	abstract void put(Object i);
	abstract Object get();
}

public class A{
	private int op = 0;
	public void put(Object i){
		s[op++] = i;
		om++;
	}
	public Object get(){
		om--;
		return s[--op];
	}
}
```
### 1.2.2 `interface-implements`
## 1.3 组合性
# 二 线程
## 1.1 线程基础知识
### 1.1.1线程的结构：
虚拟CPU，执行的代码和处理的数据。 `run()`函数是线程体，负责具体的代码执行，当线程被初始化之后，`Java` 中系统会自动调用`run()`函数。
### 1.1.2线程的状态：
1. 新建`Thread thread = new Thread();`
1. 可运行状态`thread.start();`
2. 死亡
3. 阻塞