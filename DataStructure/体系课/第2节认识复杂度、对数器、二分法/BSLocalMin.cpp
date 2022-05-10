// 问题描述：通过二分查找的方法找到数组中的任何一个局部最小值的位置
#include<iostream>
#include<vector>
#include <time.h>
using namespace std;

// 二分查找: 本题目中默认数组有序（从小到大）
int binarysearch(vector<int> arr){
    if (arr.size() <= 1) return -1;
    
    if (arr[0]<=arr[1]){
        return 0;
    }
    if (arr[arr.size()-1]<=arr[arr.size()-2]){
        return arr.size()-1;
    }
    // 如果两端都不是最小值，则初始化左右边界后用二分法
    int L = 0;
    int R = arr.size()-1;

    while (L <= R){
        int mid = L+((R-L)>>1);
        // 注意，这里访问的是mid及其两侧的值，所以有数组越界的可能，如l=0,r=1,mid=0的情况
        // mid处为局部极小值
        // printf("%d\n",arr[mid]);
        // 仅有可能超出左边界
        if (mid-1<0)
        {
            if (arr[mid] <= arr[mid+1])
                return mid;
            else{
                L = mid + 1;
            }
        }
        else{
            if (arr[mid] <= arr[mid-1] && arr[mid] <= arr[mid+1]){
                return mid;
            }
            //mid左边有局部极小值
            else if (arr[mid] > arr[mid-1])
            {
                R = mid -1;
            }
            //mid右边有局部极小值
            else{
                L = mid + 1;
            }
        }
    }
    return -1;
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


// 寻找数组中<=value的下标（暴力搜索，默认数组中相邻位置元素不同）
vector<int> find_index(vector<int> arr){
    vector<int> index;
    if (arr.size() <= 1) return index;
    if (arr[0]<=arr[1]){
        index.push_back(0);
    }
    if (arr[arr.size()-1]<=arr[arr.size()-2]){
        index.push_back(arr.size()-1);
    }
    
    for (int i = 1; i <arr.size()-1; i++){
       if (arr[i]<=arr[i+1] && arr[i]<=arr[i-1]){
           index.push_back(i);
       }
    }
    // 返回所有可能的下标
    return index;
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
            vector<int> index_arr = find_index(arr);
            int index = binarysearch(arr);
            if (index==-1){
               if (index_arr.size()!=0) {
                   succeed = false;
                   printArray(arr);
                   printArray(index_arr);
                   printf("%d\n",index);
               }  
            }
            else{
                vector<int>::iterator iElementFound;
                iElementFound = find(index_arr.begin(), index_arr.end(), index);
                //如果我的方法返回值不在暴力法中 
                if (iElementFound == index_arr.end()) {
                    succeed = false;
                    printArray(arr);
                    printArray(index_arr);
                    printf("%d\n",index);
                }
            }
		}
        printf(succeed ? "Nice!\n" : "Fucking fucked!\n");    
}

int main(){
    test();
    // 随机生成的数组很有可能相邻元素相同，所以稍微与课上的代码不同，认为a[i]<=a[i+1]且a[i]<=a[i-1]的也是局部最小值
    // vector<int> a = {-67,-67,-63,-39,-1,-96,86};
    // printArray(a);
    // printf("%d\n", binarysearch(a));
    // printArray(find_index(a));
}