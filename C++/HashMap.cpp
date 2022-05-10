#include<iostream>
#include<vector>
#include <unordered_map>
#include <iterator>
#include <time.h>
using namespace std;

// 利用异或运算求解KM问题
// 建立移位运算（如0100）与t数组下标（0100是第2位为1，所以0100要映射为2）之间映射的哈希表
void creatmap(unordered_map<int,int> &Hashmap){
    for (int i = 0; i < 32; i++){
       Hashmap[1 << i] = i;
       // 另一种插入方法
       // Hashmap.insert(make_pair(1 << i,i));
    }
}

int main(){
    unordered_map<int,int> Hashmap;
    if (Hashmap.empty()){
        creatmap(Hashmap);
    }
    
    //测试哈希表是否构造完成的代码
    for (auto it = Hashmap.begin(); it != Hashmap.end(); ++it) {
        printf("key = %d: value = %d\n",it->first, it->second);
    }
    // find返回结果为一个迭代器，需要正确使用
    printf("%d\n",Hashmap.find(4)->second);
    // 由于Hashmap.begin()这类迭代器不属于随机迭代器，因此，只能用++或者advance辅助函数来移动迭代器
    auto it_test = Hashmap.begin();
    advance(it_test, 2);
    for (auto it = it_test; it != Hashmap.end(); ++it) {
        printf("key = %d: value = %d\n",it->first, it->second);
    }
    // 等价于
    for (auto it = ++(++Hashmap.begin()); it != Hashmap.end(); ++it) {
        printf("key = %d: value = %d\n",it->first, it->second);
    }
}