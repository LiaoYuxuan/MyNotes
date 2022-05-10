#include<iostream>
#include<vector>
#include <unordered_map>
#include <unordered_set>
#include <time.h>
using namespace std;

// 利用异或运算求解奇偶个数问题
// 简单版本：数组中只有一种数的个数是奇数次，其他数字均为偶数次，找出只有奇数次数字的那个数
int exor_oddity(vector<int> arr){
    int exor = 0;
    for (auto it = arr.begin(); it != arr.end(); it++){
        exor ^= *it;
    }
    return exor;
}

// 对数器
int comparator(vector<int> arr){
    unordered_map<int,int> Hashmap;
    for (auto it = arr.begin(); it != arr.end(); it++){
        // 已存在
        if (Hashmap.find(*it) != Hashmap.end())    Hashmap[*it] +=1;
        else    Hashmap[*it] =1;
    }
    for (auto it = Hashmap.begin(); it != Hashmap.end(); it++) {
        if (it->second % 2 == 1)    return it->first;  
    }
    return -1;
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
vector<int> generateRandomArray(int maxKinds, int maxValue, int odd, int even){
        // 首先生成数据的种数（最少2种，最多maxKinds种）
        int kinds = (rand()%(maxKinds-2+1))+2;
        vector<int> arr;
        // vector<int> arr(1*odd+(kinds-1)*even);
        // 然后生成对应kinds个不同的数据，利用哈希集合判断是否重复
        unordered_set<int> Hashset;
        for (int i = 0; i < kinds; i++){
            int num = 0;
            do{
                num = (rand()%(2*maxValue+1))-maxValue;  
                // 为了避免下一次产生的数据还重复，用while结构
            } while (Hashset.find(num) != Hashset.end());
            Hashset.emplace(num);
        }
        // 先将奇数个相同的数放进去, 这里有问题，先前生成了指定大小vector后就不能用push_back了
        for (int i = 0; i < odd; i++){
            arr.push_back(*(Hashset.begin()));
        }
        // 再将偶数个相同的其他数放进去
        for (auto it = ++Hashset.begin(); it != Hashset.end(); it++){
            for (int i = 0; i < even; i++){
                arr.push_back(*it);
            }
        }
        // printArray(arr);
        //随机交换i和j处的值，打乱数组
        for (int i = 0; i < arr.size(); i++){
            int j = rand()%(arr.size());
            swap(arr, i, j);
        }
        return arr;
}



// 测试使用
void test(){
        // 最多有多少种数字
        int maxKinds = 10;
        // 每个数字的范围
        int maxValue = 30;
		int testTime = 500000;
        // K和M的最大范围
		int maxOE = 10;
        bool succeed = true;
        //srand()函数产生一个以当前时间开始的随机种子
        srand((unsigned)time(NULL));
		for (int i = 0; i < testTime; i++) {
            int odd, even;
            do{
               odd = (rand()%maxOE)+1;
            } while (odd%2 == 0);
            do{
               even = (rand()%maxOE)+1;
            } while (even%2 == 1);
            vector<int> arr = generateRandomArray(maxKinds, maxValue, odd, even);
			if (exor_oddity(arr) != comparator(arr)) {
				succeed = false;
				printArray(arr);
                printf("%d\n", exor_oddity(arr));
                printf("%d\n", comparator(arr));
				break;
			}
		}
        printf(succeed ? "Nice!\n" : "Fucking fucked!\n");    
}

int main(){
    test();
}