// 问题描述：通过二分查找的方法寻找有序数组中>=value的最左位置(注意：原数组中可以没有一样的value)
#include<iostream>
#include<vector>
#include <time.h>
using namespace std;

// 二分查找: 本题目中默认数组有序（从小到大）
int binarysearch(vector<int> arr, int value){
    if (arr.size() == 0) return -1;
    // 初始化左右边界
    int L = 0;
    int R = arr.size()-1;
    int index = -1;
    while (L <= R){
        // 这种方式还避免了奇偶整除的问题，如(5-1)>>1=2,(5-2)>>1=1
        int mid = L+((R-L)>>1);
        // L    mid    R
        // L   R
        //         L   R
        if (arr[mid] >= value){
            index = mid;
            R = mid - 1;
        }
        else{
            L = mid + 1;
        }
    }
    return index;
}

// 对数器
void comparator(vector<int> &arr){
    // 自带的排序算法，默认从小到大
    sort(arr.begin(), arr.end());
}

// 生成随机数组
vector<int> generateRandomArray(int maxSize, int maxValue){
		// rand()返回0到最大随机数的任意整数
        // (rand()%(b-a))+a: 获得[a,b)的随机整数
        // (rand()%(b-a+1))+a: 获得[a,b]的随机整数
        // (rand()%(b-a))+a+1: 获得(a,b]的随机整数

        //产生[0,maxSize-1]大小范围的数组
        vector<int> arr((rand()%(maxSize-0))+0);
        for (auto it = arr.begin(); it != arr.end(); it++){
            //产生[-maxValue, maxValue]之间的随机整数
            *it = (rand()%(2*maxValue+1))-maxValue;
        }
        return arr;
}

// 复制一个数组
vector<int> copyArray(vector<int> arr){
    // copy前需要提前申请好内存
    vector<int> copy_arr(arr.size());
    copy(arr.begin(), arr.end(), copy_arr.begin());
    return copy_arr;
}

// 寻找数组中>=value的下标
int left(vector<int> arr, int value){
    for (int i = 0; i < arr.size(); i++)
    {
        // 由于默认从小到大，那找最小匹配位置从左边开始找即可
       if (arr[i]>=value)
       {
           return i;
       }
    }
    // 数组中无该数时，返回-1
    return -1;
}

// 顺序输入数组的函数
vector<int> inputArray(){
    vector<int> arr;
    int len, tmp;
    printf("Size of array: ");
    scanf("%d", &len);
    for (int i = 0; i < len; i++)
    {
        scanf("%d", &tmp);
        // 避免初始化数组大小
        arr.push_back(tmp);
    }
    return arr;
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

// 测试使用
void test(){
		int testTime = 500000;
		int maxSize = 100;
		int maxValue = 100;
		bool succeed = true;
        //srand()函数产生一个以当前时间开始的随机种子
        srand((unsigned)time(NULL));
		for (int i = 0; i < testTime; i++) {
            vector<int> arr = generateRandomArray(maxSize, maxValue);
            int value = (rand()%(2*maxValue+1))-maxValue;
            comparator(arr);
            if (!(left(arr, value) == binarysearch(arr, value)))
            {
                succeed = false;
                printArray(arr);
                printf("%d\n",value);
                printf("%d\n",left(arr, value));
                printf("%d\n",binarysearch(arr, value));
            }
		}
        printf(succeed ? "Nice!\n" : "Fucking fucked!\n");    
}

int main(){
    test();
}