#include<iostream>
#include<vector>
#include <unordered_map>
#include <unordered_set>
#include <time.h>
using namespace std;

// 利用异或运算求解奇偶个数问题
// 进阶版本：数组中只有两种数的个数是奇数次，其他数字均为偶数次，找出只有奇数次数字的两个个数
// 由于要返回多个值，放在传入参数中比较好
void exor_oddity(vector<int> arr, int &num_1, int &num_2){
    // 第一步：对所有元素求异或
    int exor = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it){
        exor ^= *it;
    }
    // 第二步：找出exor中最右边的1
    int right = exor & (-exor);
    // 第三步：只对数组中最右边1下标对应元素为1的数求异或
    int onlyone = 0;
    for (auto it = arr.begin(); it != arr.end(); ++it){
        if ( ((*it) & right) != 0)    onlyone ^= *it;
    }
    num_1 = min(onlyone, exor ^ onlyone);
    num_2 = max(onlyone, exor ^ onlyone);

}

// 对数器
void comparator(vector<int> arr, int &num_1, int &num_2){
    unordered_map<int,int> Hashmap;
    for (auto it = arr.begin(); it != arr.end(); ++it){
        // 已存在
        if (Hashmap.find(*it) != Hashmap.end())    Hashmap[*it] +=1;
        else    Hashmap[*it] =1;
    }
    vector<int> tmp;
    for (auto it = Hashmap.begin(); it != Hashmap.end(); ++it) {
        if (it->second % 2 == 1)    tmp.push_back(it->first);  
    }
    num_1 = min(tmp[0], tmp[1]);
    num_2 = max(tmp[0], tmp[1]);
}

// 交换函数
void swap(vector<int> &arr, int index1, int index2){
    int tmp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = tmp;
}

// 顺序输出数组的函数
void printArray(vector<int> arr){
    // 使用迭代器的方式访问vector，这里arr.begin()和arr.end()都是指针, auto是vector<int>::iterator的简写形式
    for (auto it = arr.begin(); it != arr.end(); it++)
    {
        printf("%d ",*it);
    }
    cout << endl;
}

// 生成符合要求的随机数组
vector<int> generateRandomArray(int maxKinds, int maxValue, int odd_1, int odd_2, int even){
        // 首先生成数据的种数（最少2种，最多maxKinds种）
        int kinds = (rand()%(maxKinds-2+1))+2;
        vector<int> arr;
        // 然后生成对应kinds个不同的数据，利用哈希集合判断是否重复
        unordered_set<int> Hashset;
        for (int i = 0; i < kinds; ++i){
            int num = 0;
            do{
                num = (rand()%(2*maxValue+1))-maxValue;  
                // 为了避免下一次产生的数据还重复，用while结构
            } while (Hashset.find(num) != Hashset.end());
            Hashset.emplace(num);
        }
        // 先将奇数个相同的数放进去, 这里有问题，先前生成了指定大小vector后就不能用push_back了
        for (int i = 0; i < odd_1; ++i){
            arr.push_back(*(Hashset.begin()));
        }
        for (int i = 0; i < odd_2; ++i){
            arr.push_back(*(++Hashset.begin()));
        }
        // 再将偶数个相同的其他数放进去
        // 对循环控制变量 i，要养成写++i、不写i++的习惯。
        for (auto it = ++(++Hashset.begin()); it != Hashset.end(); ++it){
            for (int i = 0; i < even; i++){
                arr.push_back(*it);
            }
        }
        // printArray(arr);
        //随机交换i和j处的值，打乱数组
        for (int i = 0; i < arr.size(); ++i){
            int j = rand()%(arr.size());
            swap(arr, i, j);
        }
        return arr;
}



// 测试使用
void test(){
        // 最多有多少种数字
        int maxKinds = 20;
        // 每个数字的范围
        int maxValue = 30;
		int testTime = 500000;
        // K和M的最大范围
		int maxOE = 10;
        bool succeed = true;
        //srand()函数产生一个以当前时间开始的随机种子
        srand((unsigned)time(NULL));
		for (int i = 0; i < testTime; ++i) {
            int odd_1, odd_2, even;
            do{
               odd_1 = (rand()%maxOE)+1;
            } while (odd_1%2 == 0);
            do{
               odd_2 = (rand()%maxOE)+1;
            } while (odd_2%2 == 0);
            do{
               even = (rand()%maxOE)+1;
            } while (even%2 == 1);
            vector<int> arr = generateRandomArray(maxKinds, maxValue, odd_1, odd_2, even);
            int num1, num2;
            exor_oddity(arr, num1, num2);
            int num3, num4;
            comparator(arr, num3, num4);
			if (num1!=num3||num2!=num4) {
				succeed = false;
				printArray(arr);
                printf("%d\n", num1);
                printf("%d\n", num2);
                printf("%d\n", num3);
                printf("%d\n", num4);
				break;
			}
		}
        printf(succeed ? "Nice!\n" : "Fucking fucked!\n");    
}

int main(){
    test();
}