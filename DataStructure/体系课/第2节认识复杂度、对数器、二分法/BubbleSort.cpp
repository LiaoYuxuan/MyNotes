#include<iostream>
#include<vector>
#include <time.h>
using namespace std;

// 交换函数
void swap(vector<int> &arr, int index1, int index2){
    int tmp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = tmp;
}

// 冒泡排序
// 从0~N-1中，相邻比较大小，大的往后放
// 从0～N-2中，相邻比较大小，大的往后放，依次类推
void bubblesort(vector<int> &arr){
    if (arr.size() < 2) return;
    for (int i = arr.size()-1; i >= 0; i--)
    {
       for (int j = 0; j <= i-1; j++)
       {
           if (arr[j]>arr[j+1])
           {
               swap(arr, j, j+1);
           }
       }
    }   
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

// 比较两个数组是否相同
bool isEqual(vector<int> arr1, vector<int> arr2){
    if ((arr1.size()==0 && arr2.size()!=0) || (arr1.size()!=0 && arr2.size()==0) || (arr1.size()!=arr2.size())){
        return false;
    }
    if (arr1.size()==0 && arr2.size()==0){
        return true;
    }
    for (int i = 0; i < arr1.size(); i++)
    {
        if (arr1[i] != arr2[i])
        {
           return false;
        }
    }
    return true;
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
            vector<int> arr1 = copyArray(arr);
            vector<int> arr2 = copyArray(arr);
            bubblesort(arr1);
            comparator(arr2);
			if (!isEqual(arr1, arr2)) {
				succeed = false;
				printArray(arr);
				break;
			}
		}
        printf(succeed ? "Nice!\n" : "Fucking fucked!\n");    
}

int main(){
    test();
}