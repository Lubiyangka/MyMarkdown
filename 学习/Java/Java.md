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
# 三 基础
### `double`和`float`的区别
`double`是双精度`float`是单精度的，一般小数默认是`double`
```java
	double x = 1.0;
	//以下是错误的
	float y = 1.0;
	//应该是
	float z = 1.0f;
	//或
	float i = 1.0F;
```
# 概述
## 1.1 Java介绍
Java的前身是`Oak`。最早的面向对象语言是Simula然后是Smalltalk
`Java`的区分：
- 企业版J2EE
- 标准版J2SE
- 微型版J2ME
区别于C和C++无需指针和多重继承，通过垃圾自动回收机制简化沉痼内存管理。Java源程序被编译器百衲衣之后会转为字节码的目标程序”编写一次，到处运行“
### 1.1.1Java运行机制
使用java编辑工具编写.java文件，通过Java编译器编译.class文件。
$$.java文件 \rightarrow{.class文件}$$
### 1.1.2Java虚拟机`JVM`：
[Java JVM 运行机制及基本原理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/25713880)
含义：右软件技术模拟出来的计算机运行的一个虚拟计算机。
工作原理：java编译器将.java文件编译成.class文件之后，为了在不同系统上运行，需要开发对应的解释器，通过解释器，在JVM上可以运行.class文件。
一般使用最广泛的是由`OracleJDK`或`OpenJDK` 中默认的HotSpot虚拟机。
### 1.1.3 `JCP`与`JSR`
## 1.2 应用
### 1.2.1 `web`应用
`Servlet` `Applet` `JavaBean`
# 二 标识符和数据类型
结构说明：package导包使用，import类似"include"的作用，interface表示接口
```Java
import java.util.Random;
//import需在类定义之前
//public类型的class一个文件中只能有一个,Thread就是一个非public类型的class
public class Main {  
    public static void main(String[] args) {  
        ...
    }  
}  
  
class HorseThread extends Thread {  
    ...
    ...
}
```
Java使用的是Unicode其前128个字符与ASCII相同。
数据类型：
- boolean布尔类型
- char字符类型
- float浮点型
- double双精度

|  | float | double |
| --- | --- | --- |
| 最大值 | Float.MAX_VALUE | Double.MAX_VALUE | 
| 正无穷大 | Float.POSITIVE_INFINITY | Double.POSITIVE_INFINITY | 
# 三 面向对象
## 3.1 三大特性
继承，封装，多态。
## 3.2 类的定义
### 3.2.1 类
示例：
```Java
public class Test extends ParentClass implements ImplementClass1 {
	private Double price;
	private Long id;
	private Boolean tag;
	//构造函数
	public Test(){
		...	
	}
	
	//析构函数
	public ~Test(){
		...
	}
	public void Function1(int i, char y){
		...
	}
}

[访问符][修饰符] class 类名 [<类型参数列表>] [extends 父类名] [implements接口名列表] {
	[成员变量列表]
	[成员方法列表] 
}
```
1. 访问权限的修饰符：`public`和`private`和`protected`

| 类型 | 无修饰符 | private | public | protected |
| ---- | ---- | ---- | ---- | ---- |
| 同包同类 | &#10004 |&#10004 |&#10004 |&#10004 |
| 同包子类 | &#10004| | &#10004|&#10004 |
| 同包非子类 | &#10004| | &#10004|&#10004 |
| 不同包子类 | | |&#10004 |&#10004 |
| 不同包非子类 | | | &#10004|  |
2. 存储方式修饰符`static`：修饰类的属性或方法，变为静态属性或方法。可以被类的所有对象所共享。
3. `final`：
	1. 修饰类：修饰的类不能再派生子类，表示该类不能被继承，fianl类的所有成员方法都会被隐式的指定为fianl方法。
	2. 修饰方法：可以锁定防止子类修改含义。
	3. 修饰变量：修饰基础数据类型的变量，其值初始化之后不能被改变；修饰引用类型变量，指向对象之后就不能改变指向了。
4. abstract用来修饰类或方法，表示被修饰的成分是抽象的，抽象方法给出函数原型。含有抽象方法的类是抽象类。
5. this是指本类
6. super是指父类
### 3.2.2成员变量
`[访问符][修饰符列表] 变量类型 变量名 [=初始值];`
1. 访问符：
	1. public
	2. protected
	3. private
	4. 缺省
2. 修饰符列表
	1. static：表示静态变量
	2. final：表示常量
	3. transient：表示变量不可持续化
	4. volatile：表示变量是一个可能同时被并行运行中的几个线程所控制和修改的变量
### 3.2.3 成员函数
`[访问符] [修饰符列表] <返回类型> 方法名([参数列表]){ 方法体 }`
1. 访问符：
	1. public
	2. protected
	3. private
	4. 缺省
2. 修饰符列表：
	1. static：不需要创建对象就可以使用，类访问
	2. final：不允许子类重写
	3. abstract：抽象方法，无函数体，fianl和abstract不能同时出现。
	4. synchronized：表示该方法时一个线程同步方法
参数列表的可变参数：只能在队尾，只能包含一个可变参数，数据类型相同
 ```Java
 public static int sum(int x, int ... a){  
    int sum = x;  
    for(int i : a){  
        sum += i;  
    }  
    return sum;  
  
}  
public static void main(String[] args) {  
    int x = sum(2,1,1,1,1,1,1,1,1,1,1);  
    System.out.println(x);  
}
```
### 3.2.4 实例化对象
步骤：$对象的引用 + 对象的实例化 \rightarrow{对象的说明}$
对象的引用：不指向任何的内存空间
对象的实例化：申请相应的空间，建立首地址引用。
```Java
Point p;
p = new Point();
Point p = new Point();
```
对象的赋值
```Java
	Test a = new Test(1, "hello");  
	System.out.println(a.getId()+" "+a.getName());  
	Test b = a;  
	b.setId(2);  
	b.setName("world");  
	System.out.println(a.getId()+" "+a.getName());  
	System.out.println(b.getId()+" " +b.getName());
```
>输出:
>	1 hello
	   2 world
	   2 world
### 3.2.5 继承
子类可以直接new给父类，父类对象需要（子类名）强制转化为子类对象。
### 3.2.6 接口
interface允许定义类的方法名，自变量列表和返回类型，但不包含方法主体，都是抽象函数，是特殊的抽象类。不用于继承的是，一个类可以implement多个接口。
### 3.2.7 组合性
Java支持一个类里声明了一个类，这个类是**内部类**，其所在的类是**外部类**，根据内部类所在的位置可以分为：
1. 成员类：成员类不能与外部类重名，访问权限也是4种。成员类可以fianl，然后就不能被继承；可以是abstract，但是需要被其他的内部类继承或实现。
	1. 非静态成员类：没有static修饰符，不能定义静态的变量和函数。非静态成员类可以访问外部类的所有成员，访问时，不重名时，可以直接用成员名称访问；访问外部类成员需要`外部类.this.外部类成员变量名方法名(参数列表)`
```Java  
public class Main {  
    //外部类成员变量重名  
    private int id;  
    //外部类成员变量不重名  
    private String address;  
    public Main(int id,String address){  
        this.id = id;  
        this.address = address;  
    }  
  
    public static void main(String[] args) {  
        Main main = new Main(1, "xian");  
        main.print(3, "dafd");  
//        Main.Test test = main.new Test(2,"hlq");  
//        System.out.println(test.toString());  
    }  
  
    public void  print(int id, String name){  
        Test test = new Test(id, name);  
        System.out.println(test.toString());  
    }  
  
    public class Test {  
        private int id;  
        private String name;  
        public Test(int id, String name){  
            this.id = id;  
            this.name = name;  
        }  
        public String toString() {  
            return "this is Main id = " + Main.this.id + " this is Main address = " + Main.this.address +  
                    " this is test id = " + this.id + " this is test name = " + this.name;  
        }  
    }  
}
```
3. 局部类：定义在方法或作用域中
4. 匿名类：不能是abstract，static，只能是final，所有的方法和变量都是final
```Java
public class Main {  
  
  
    public static void main(String[] args) {  
        Main main = new Main();  
        main.show();  
    }  
    public void show(){  
        Out out = new Out(){  
            @Override  
            void show() {  
                super.show();  
                System.out.println("Main.show");  
            }  
        };  
        out.show();  
    }  
}  
class Out{  
    void show(){  
        System.out.println("Out.show");  
    }  
}
```
# 表达式&流程控制
## 运算符：
### 算数运算符
`%`对double类型数据取余时，整数部分取余，然后小数部分保留。
```Java
	double i = 12.8 % 4;  
	System.out.println(i);
```
输出：0.8000000000000007
### 逻辑运算符
与`&`，短路与`&&`，或`|`，短路或`||`，非`!`，异或`^`
短路与/或--与/或的区别在于短路类型当前部分判断已经可以得出结果是，就停止后续的判断。
```Java
public static void main(String[] args) {  
//        double i = 12.8 % 4;  
//        System.out.println(i);  
        // 验证短路与  
        System.out.println("短路与：");  
        boolean result = A() && B();  
        System.out.println("---------------");  
        // 验证短路或  
        System.out.println("短路或：");  
        result = A() || B();  
        System.out.println("---------------");  
        // 验证与  
        System.out.println("与：");  
        result = A() & B();  
        System.out.println("---------------");  
        // 验证或  
        System.out.println("或：");  
        result = A() | B();  
    }  
    private static boolean A() {  
        System.out.println("enter A");  
        boolean result = false;  
        // 生成随机的true和false  
        Random random = new Random();  
        result = random.nextBoolean();  
        // 输出返回的boolean类型是true还是false  
        System.out.println("A:"+result);  
        return result;  
    }  
  
    private static boolean B() {  
        System.out.println("enter B");  
        boolean result = false;  
        // 生成随机的true和false  
        Random random = new Random();  
        result = random.nextBoolean();  
        // 输出返回的boolean类型是true还是false  
        System.out.println("B:"+result);  
        return result;  
    }
```
`instanceof`实例化运算符
# 数组，字符串，容器
String不可变
StringBuffer可变
```Java
	String n = "abc";  
	StringBuffer m = new StringBuffer();  
	m.append("s");  
	m.append("afads");  
	System.out.println(m);
```
StringBuffer的length和capacity的区别：
1. StringBuffer的的初始大小为（16+初始字符串长度）即capacity=16+初始字符串长度
2. 一旦length大于capacity时，capacity便在前一次的基础上加1后倍增；
length=1;capacity=17;//初始长度
length=5;capacity=17;//
length=17;capacity=17;//
length=18;capacity=(capacity+1)\*2=36;//第一次倍增
length=37;capacity=(capacity+1)\*2=74;//第二次倍增
容器:List，Set，Queue，Map
`Collection`
```Java
public class Main {  
    private Long id;  
    private String name;  
    private static Collection<Object> collection = new ArrayList<>();  
    public Main(long id, String name){  
        this.id = id;  
        this.name = name;  
    }  
    public static void main(String[] args) {  
        collection.add(new Main(3L, "haha"));  
        for (Object o : collection) {  
            Main main = (Main) o;  
            System.out.println(main.id + " " + main.name);  
        }  
    }
}
```
`Iterator`
hasnext()判断序列是否还有元素，next()下一个,remove()删除当前元素
`List`
sort(List),升序,reverse(List)降序，
```Java
public class Main implements Comparable{  
    private Long id;  
    private String name;  
    private static List list = new ArrayList<>();  
    public Main(long id, String name){  
        this.id = id;  
        this.name = name;  
    }  
  
    @Override  
    public String toString() {  
        return "Main{" +  
                "id=" + id +  
                ", name='" + name + '\'' +  
                '}';  
    }  
  
    @Override  
    public int compareTo(Object o) {  
        Main main = (Main) o;  
//        return (int) (this.id - main.id);  
        return this.name.compareTo(main.name);  
    }  
  
    public static void main(String[] args) {  
        Main.list.add(new Main(1, "computer"));  
        Main.list.add(new Main(2,"software"));  
        Main.list.add(new Main(4, "qw"));  
        Main.list.add(new Main(5,"as"));  
        Main.list.add(new Main(3, "gh"));  
        Main.list.add(new Main(6,"bn"));  
        System.out.println(Main.list);  
        Collections.sort(Main.list);  
        System.out.println(Main.list);  
    }  
}
```
# 异常处理
常见的公共异常：
1. `ArithmeticException`除数为0
2. `NullPointerException`对象未初始化就调用方法或访问对象。
4. `NegativeException`数组创建时，元素个数为负数
5. `ArrayIndexOutOfBoundsException`数组越界
6. `ArrayStoreException`程序试图存取数组中的错误数据类型
7. `FileNotFoundException`存取一个不存在的文件
8. `IOException`I/0错误
## 异常分类
1. 错误异常
2. 程序可处理异常
## 处理方法
处理方法：
1. 捕获异常
2. 转移异常
### 捕获异常
try-catch-finally结构处理
```Java
public static void throwException(){  
    int i = 0;  
    String[] str = {"Curry", "James", "John", "Niko", "Simple"};  
    while(i<6){  
        try{  
            System.out.print(str[i]);  
        }catch (ArrayIndexOutOfBoundsException e){  
            System.out.println("\nResetting Index Value");  
        }catch (Exception e){  
            System.out.println(e.toString());  
        }finally {  
            System.out.print("-->");  
        }  
        i++;  
    }  
}
```
### 转移异常
```Java
public static void throwSome() throws Exception{  
    int i = 0;  
    String[] str = {"Curry", "James", "John", "Niko", "Simple"};  
    while(i<6){  
        System.out.println(str[i]);  
        i++;  
    }  
}
public static void main(String[] args){  
    //throwException();
    //System.out.println("\n\n");  
    try{  
        throwSome();  
    }catch (Exception e){  
        System.out.println(e.toString());  
    }  
}
```
