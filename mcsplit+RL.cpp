/*
 *Copyright(C) 2019
 *Yanli Liu(yanlil2008@163.com)

 *References:
 *Yanli Liu, Chu-Min Li, Jiang Hua, Kun He,
 *"A Learning based Branch and Bound for Maximum Common Subgraph related Problems",
 *In AAAI20

 *This program is based on McSplit (Ciaran McCreesh, Patrick Prosser and James Trimble: A Partitioning Algorithm for Maximum Common Subgraph Problems.)
  *you can download McSplit code from https://github.com/jamestrimble/ijcai2017-partitioning-common-subgraph

 *This program is for MCS problem and MCCS problem based on a learning of branching vertices.

 *This program is free software. you can redistribute it and/or modify it for research.
*/
#include "graph.h"

#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

#include <argp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define VSCORE
//#define DEBUG
#define Best
using std::vector;
using std::cout;
using std::endl;

static void fail(std::string msg) {
    std::cerr << msg << std::endl;
    exit(1);
}


enum Heuristic { min_max, min_product };

/*******************************************************************************
                             Command-line arguments
*******************************************************************************/

static char doc[] = "Find a maximum clique in a graph in DIMACS format\vHEURISTIC can be min_max or min_product";
static char args_doc[] = "HEURISTIC FILENAME1 FILENAME2";
static struct argp_option options[] = {
    {"quiet", 'q', 0, 0, "Quiet output"},
    {"verbose", 'v', 0, 0, "Verbose output"},
    {"dimacs", 'd', 0, 0, "Read DIMACS format"},
    {"lad", 'l', 0, 0, "Read LAD format"},
    {"connected", 'c', 0, 0, "Solve max common CONNECTED subgraph problem"},
    {"directed", 'i', 0, 0, "Use directed graphs"},
    {"labelled", 'a', 0, 0, "Use edge and vertex labels"},
    {"vertex-labelled-only", 'x', 0, 0, "Use vertex labels, but not edge labels"},
    {"big-first", 'b', 0, 0, "First try to find an induced subgraph isomorphism, then decrement the target size"},
    {"timeout", 't', "timeout", 0, "Specify a timeout (seconds)"},
    { 0 }
};

static struct {
    bool quiet;
    bool verbose;
    bool dimacs;
    bool lad;
    bool connected;
    bool directed;
    bool edge_labelled;
    bool vertex_labelled;
    bool big_first;
    Heuristic heuristic;
    char *filename1;
    char *filename2;
    int timeout;
    int arg_num;
} arguments;

static std::atomic<bool> abort_due_to_timeout;

void set_default_arguments() {
    arguments.quiet = false;
    arguments.verbose = false;
    arguments.dimacs = false;
    arguments.lad = false;
    arguments.connected = false;
    arguments.directed = false;
    arguments.edge_labelled = false;
    arguments.vertex_labelled = false;
    arguments.big_first = false;
    arguments.filename1 = NULL;
    arguments.filename2 = NULL;
    arguments.timeout = 0;
    arguments.arg_num = 0;
}

static error_t parse_opt (int key, char *arg, struct argp_state *state) {
    switch (key) {
        case 'd':
            if (arguments.lad)
                fail("The -d and -l options cannot be used together.\n");
            arguments.dimacs = true;
            break;
        case 'l':
            if (arguments.dimacs)
                fail("The -d and -l options cannot be used together.\n");
            arguments.lad = true;
            break;
        case 'q':
            arguments.quiet = true;
            break;
        case 'v':
            arguments.verbose = true;
            break;
        case 'c':
            if (arguments.directed)
                fail("The connected and directed options can't be used together.");
            arguments.connected = true;
            break;
        case 'i':
            if (arguments.connected)
                fail("The connected and directed options can't be used together.");
            arguments.directed = true;
            break;
        case 'a':
            if (arguments.vertex_labelled)
                fail("The -a and -x options can't be used together.");
            arguments.edge_labelled = true;
            arguments.vertex_labelled = true;
            break;
        case 'x':
            if (arguments.edge_labelled)
                fail("The -a and -x options can't be used together.");
            arguments.vertex_labelled = true;
            break;
        case 'b':
            arguments.big_first = true;
            break;
        case 't':
            arguments.timeout = std::stoi(arg);
            break;
        case ARGP_KEY_ARG:
            if (arguments.arg_num == 0) {
                if (std::string(arg) == "min_max")
                    arguments.heuristic = min_max;
                else if (std::string(arg) == "min_product")
                    arguments.heuristic = min_product;
                else
                    fail("Unknown heuristic (try min_max or min_product)");
            } else if (arguments.arg_num == 1) {
                arguments.filename1 = arg;
            } else if (arguments.arg_num == 2) {
                arguments.filename2 = arg;
            } else {
                argp_usage(state);
            }
            arguments.arg_num++;
            break;
        case ARGP_KEY_END:
            if (arguments.arg_num == 0)
                argp_usage(state);
            break;
        default: return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };

/*******************************************************************************
                                     Stats
*******************************************************************************/

unsigned long long nodes{ 0 };
unsigned long long cutbranches{0};
unsigned long long conflicts=0;
clock_t bestfind;
unsigned long long bestnodes=0,bestcount=0;
int dl=0;
/*******************************************************************************
                                 MCS functions
*******************************************************************************/

struct VtxPair {
    int v;
    int w;
    VtxPair(int v, int w): v(v), w(w) {}
};

struct Bidomain {
    int l,        r;        // start indices of left and right sets
    int left_len, right_len;
    bool is_adjacent;
    Bidomain(int l, int r, int left_len, int right_len, bool is_adjacent):
            l(l),
            r(r),
            left_len (left_len),
            right_len (right_len),
            is_adjacent (is_adjacent) { };
};

void show(const vector<VtxPair>& current, const vector<Bidomain> &domains,
        const vector<int>& left, const vector<int>& right)
{
    cout << "Nodes: " << nodes << std::endl;
    cout << "Length of current assignment: " << current.size() << std::endl;
    cout << "Current assignment:";
    for (unsigned int i=0; i<current.size(); i++) {
        cout << "  (" << current[i].v << " -> " << current[i].w << ")";
    }
    cout << std::endl;
    for (unsigned int i=0; i<domains.size(); i++) {
        struct Bidomain bd = domains[i];
        cout << "Left  ";
        for (int j=0; j<bd.left_len; j++)
            cout << left[bd.l + j] << " ";
        cout << std::endl;
        cout << "Right  ";
        for (int j=0; j<bd.right_len; j++)
            cout << right[bd.r + j] << " ";
        cout << std::endl;
    }
    cout << "\n" << std::endl;
}

bool check_sol(const Graph & g0, const Graph & g1 , const vector<VtxPair> & solution) {
    return true;
    vector<bool> used_left(g0.n, false);
    vector<bool> used_right(g1.n, false);
    for (unsigned int i=0; i<solution.size(); i++) {
        struct VtxPair p0 = solution[i];
        if (used_left[p0.v] || used_right[p0.w])
            return false;
        used_left[p0.v] = true;
        used_right[p0.w] = true;
        if (g0.label[p0.v] != g1.label[p0.w])
            return false;
        for (unsigned int j=i+1; j<solution.size(); j++) {
            struct VtxPair p1 = solution[j];
            if (g0.adjmat[p0.v][p1.v] != g1.adjmat[p0.w][p1.w])
                return false;
        }
    }
    return true;
}

int calc_bound(const vector<Bidomain>& domains) {
    int bound = 0;
    for (const Bidomain &bd : domains) {
        bound += std::min(bd.left_len, bd.right_len);
    }
    return bound;
}

int selectV(const vector<int>& arr,const vector<int> & lgrade, int start_idx, int len) {
    //int min_v = INT_MAX;
    int max_g=-1;
    int max_v=-1;
    for (int i=0; i<len; i++){
        if (lgrade[arr[start_idx+i]]>max_g){
            max_v=arr[start_idx+i];
            max_g =lgrade[max_v];
        }
        else if(lgrade[arr[start_idx+i]]==max_g){
            if (arr[start_idx + i] < max_v)
                max_v=arr[start_idx + i];
         }
    }
    return max_v;
}

int select_bidomain(const vector<Bidomain>& domains, const vector<int> & left,const vector<int> & lgrade,
        int current_matching_size)
{
    // Select the bidomain with the smallest max(leftsize, rightsize), breaking
    // ties on the smallest vertex index in the left set
    int min_size = INT_MAX;
    int min_tie_breaker = INT_MAX;
    int tie_breaker;
    unsigned int i;  int len;
    int best = -1;
    for (i=0; i<domains.size(); i++) {
        const Bidomain &bd = domains[i];
        if (arguments.connected && current_matching_size>0 && !bd.is_adjacent) continue;
            len = arguments.heuristic == min_max ?
                std::max(bd.left_len, bd.right_len) :
                bd.left_len * bd.right_len;//最多有这么多种连线
        if (len < min_size) {//优先计算可能连线少的情况,当domain中可能配对数最小时，选择domain中含有最大度的点那个domain
            min_size = len;//此时进行修改将度最大的点进行优化
            min_tie_breaker = selectV(left,lgrade,bd.l, bd.left_len);//序号越小度越大
            best = i;
        } else if (len == min_size) {
            tie_breaker = selectV(left,lgrade,bd.l, bd.left_len);
            if (tie_breaker < min_tie_breaker) {
                min_tie_breaker = tie_breaker;
                best = i;
            }
        }
    }
    return best;
}

// Returns length of left half of array
int partition(vector<int>& all_vv, int start, int len, const vector<unsigned int> & adjrow) {
    int i=0;
    for (int j=0; j<len; j++) {
        if (adjrow[all_vv[start+j]]) {
            std::swap(all_vv[start+i], all_vv[start+j]);
            i++;
        }
    }
    return i;
}

// multiway is for directed and/or labelled graphs
vector<Bidomain> rewardfeed(const vector<Bidomain> & d,int bd_idx,vector<VtxPair> & current, vector<int> & left,
        vector<int> & right,vector<int> & lgrade,vector<int> & rgrade, const Graph & g0, const Graph & g1, int v, int w,
        bool multiway)
{//每个domain均分成2个new_domain分别是与当前对应点相连或者不相连
    vector<Bidomain> new_d;
    new_d.reserve(d.size());
    //unsigned int old_bound = current.size() + calc_bound(d)+1;//这里的domain是已经去掉选了的点，但是该点还没纳入current所以+1
    //unsigned int new_bound(0);
    int l,r,j=-1; unsigned int i;
    int temp=0,total=0;
    for (const Bidomain &old_bd : d) {
        j++;
        l = old_bd.l;
        r = old_bd.r;
        // After these two partitions, left_len and right_len are the lengths of the
        // arrays of vertices with edges from v or w (int the directed case, edges
        // either from or to v or w)
        int left_len = partition(left, l, old_bd.left_len, g0.adjmat[v]);//将与选出来的顶点相连顶点数返回，
        int right_len = partition(right, r, old_bd.right_len, g1.adjmat[w]);//并且将相连顶点依次交换到数组前面
        int left_len_noedge = old_bd.left_len - left_len;
        int right_len_noedge = old_bd.right_len - right_len;
//这里传递v,w选取的下标到bd_idx，j用来计算当前所分domain的下标，这样就能将bound更精确地计算
        temp=std::min(old_bd.left_len,old_bd.right_len)-std::min(left_len,right_len)-std::min(left_len_noedge,right_len_noedge);
        total+=temp;

#ifdef DEBUG

        printf("adj=%d ,noadj=%d ,old=%d ,temp=%d \n",std::min(left_len,right_len),
               std::min(left_len_noedge,right_len_noedge),std::min(old_bd.left_len,old_bd.right_len),temp);
        cout<<"j="<<j<< "  idx="<<bd_idx<<endl;
        cout<<"gl="<<lgrade[v]<<" gr="<<rgrade[w]<<endl;
 #endif
        if (left_len_noedge && right_len_noedge)//new_domain存在的条件是同一domain内需要存在与该对应点同时相连，或者同时不相连的点的点
            new_d.push_back({l+left_len, r+right_len, left_len_noedge, right_len_noedge, old_bd.is_adjacent});
        if (multiway && left_len && right_len) {//不与顶点相连的顶点
            auto& adjrow_v = g0.adjmat[v];
            auto& adjrow_w = g1.adjmat[w];
            auto l_begin = std::begin(left) + l;
            auto r_begin = std::begin(right) + r;
            std::sort(l_begin, l_begin+left_len, [&](int a, int b)
                    { return adjrow_v[a] < adjrow_v[b]; });
            std::sort(r_begin, r_begin+right_len, [&](int a, int b)
                    { return adjrow_w[a] < adjrow_w[b]; });
            int l_top = l + left_len;
            int r_top = r + right_len;
            while (l<l_top && r<r_top) {
                unsigned int left_label = adjrow_v[left[l]];
                unsigned int right_label = adjrow_w[right[r]];
                if (left_label < right_label) {
                    l++;
                } else if (left_label > right_label) {
                    r++;
                } else {
                    int lmin = l;
                    int rmin = r;
                    do { l++; } while (l<l_top && adjrow_v[left[l]]==left_label);
                    do { r++; } while (r<r_top && adjrow_w[right[r]]==left_label);
                    new_d.push_back({lmin, rmin, l-lmin, r-rmin, true});
                }
            }
        } else if (left_len && right_len) {
            new_d.push_back({l, r, left_len, right_len, true});//与顶点相连的顶点 标志为Ture
        }
    }
    if(total>0){
        conflicts++;
        lgrade[v]+=total;
        if(lgrade[v]>1e9){
            for(i=0;i<lgrade.size();i++)
                lgrade[i]=lgrade[i]/1e9;
        }

        rgrade[w]+=total;
        if(rgrade[w]>1e9){
            for(i=0;i<rgrade.size();i++)
                rgrade[i]=rgrade[i]/1e9;
        }
    }
#ifdef DEBUG
  cout<<"new domains are "<<endl;
  for (const Bidomain &testd : new_d){
      l = testd.l;
      r = testd.r;
      for(j=0;j<testd.left_len;j++)
          cout<<left[l+j] <<" ";
      cout<<" ; ";
      for(j=0;j<testd.right_len;j++)
          cout<<right[r+j]<<" " ;
      cout<<endl;
  }
#endif
    return new_d;
}

// returns the index of the smallest value in arr that is >w.
// Assumption: such a value exists
// Assumption: arr contains no duplicates
// Assumption: arr has no values==INT_MAX
int selectW(const vector<int>& arr, const vector<int> & rgrade,int start_idx, int len, vector<int>& wselect) {
    int idx = -1;
    int high=INT_MIN;
#ifdef VSCORE
    for (int i=0; i<len; i++) {
        if (wselect[arr[start_idx + i]]==0){
            if(rgrade[arr[start_idx + i]]>high){
                high=rgrade[arr[start_idx + i]];
                idx=i;
            }
            else if(rgrade[arr[start_idx + i]]==high)
                if(arr[start_idx + i]<arr[start_idx + idx])
                    idx=i;

            }
    }
#endif
    return idx;
}

void remove_vtx_from_left_domain(vector<int>& left, Bidomain& bd, int v)
{
    int i = 0;
    while(left[bd.l + i] != v) i++;
    std::swap(left[bd.l+i], left[bd.l+bd.left_len-1]);//left中的值已经代表了排序，所以这样交换去除点，不会影响后面的排序
    bd.left_len--;
}

void remove_bidomain(vector<Bidomain>& domains, int idx) {
    domains[idx] = domains[domains.size()-1];
    domains.pop_back();
}

void solve(const Graph & g0, const Graph & g1,vector<int> & lgrade,vector<int> & rgrade, vector<VtxPair> & incumbent,
        vector<VtxPair> & current, vector<Bidomain> & domains,
        vector<int> & left, vector<int> & right, unsigned int matching_size_goal)
{
  //  if (abort_due_to_timeout)
   //     return;
 nodes++;
    //if (arguments.verbose) show(current, domains, left, right);

    if (current.size() > incumbent.size()) {//incumbent 现任的
        incumbent = current;
        bestcount=cutbranches+1;
        bestnodes=nodes;
        bestfind=clock();
        if (!arguments.quiet) cout << "Incumbent size: " << incumbent.size() << endl;
    }

    unsigned int bound = current.size() + calc_bound(domains);//计算相连和不相连同构数的最大可能加上当前已经同构的点数
    if (bound <=incumbent.size() || bound < matching_size_goal){//剪枝
        cutbranches++;
        return;
    }
   if (arguments.big_first && incumbent.size()==matching_size_goal)
       return;



    int bd_idx = select_bidomain(domains, left,lgrade,current.size());//选出domain中可能情况最少的下标
    if (bd_idx == -1) {  // In the MCCS case, there may be nothing we can branch on
      //  cout<<endl;
      //  cout<<endl;
      //  cout<<"big"<<endl;
        return;}
    Bidomain &bd = domains[bd_idx];

    int v = selectV(left,lgrade, bd.l, bd.left_len);//选取v值时不再是选序号最小的点，而是选grade最大的点，当grade均相同时，则选度最大的。
    remove_vtx_from_left_domain(left, domains[bd_idx], v);//从中去除了度最大的点

    // Try assigning v to each vertex w in the colour class beginning at bd.r, in turn
    int w = -1;
    vector<int> wselect(g1.n,0);
    bd.right_len--;//
    for (int i=0; i<=bd.right_len; i++) {
        int idx =selectW(right,rgrade, bd.r, bd.right_len+1, wselect);//选值仅比w大的次小点??????????
        w = right[bd.r + idx];
        wselect[right[bd.r + idx]]=1;

        // swap w to the end of its colour class
        right[bd.r + idx] = right[bd.r + bd.right_len];
        right[bd.r + bd.right_len] = w;
#ifdef DEBUG
     cout<<"v= "<<v<<" w= "<<w<<endl;
     unsigned int m;
     for(m=0;m<left.size();m++)
         cout<<left[m]<<" ";
     cout<<endl;
     for(m=0;m<right.size();m++)
         cout<<right[m]<<" ";
     cout<<endl;
#endif
        auto new_domains = rewardfeed(domains,bd_idx,current, left, right, lgrade, rgrade, g0, g1, v, w,
                arguments.directed || arguments.edge_labelled);
        current.push_back(VtxPair(v, w));
        dl++;
        solve(g0, g1,lgrade,rgrade, incumbent, current, new_domains, left, right, matching_size_goal);
        current.pop_back();
    }
    bd.right_len++;
    if (bd.left_len == 0)
        remove_bidomain(domains, bd_idx);
    solve(g0, g1,lgrade,rgrade, incumbent, current, domains, left, right, matching_size_goal);
}

vector<VtxPair> mcs(const Graph & g0, const Graph & g1,vector<int> & lgrade,vector<int> & rgrade) {
    vector<int> left;  // the buffer of vertex indices for the left partitions
    vector<int> right;  // the buffer of vertex indices for the right partitions

    auto domains = vector<Bidomain> {};

    std::set<unsigned int> left_labels;
    std::set<unsigned int> right_labels;
    for (unsigned int label : g0.label) left_labels.insert(label);
    for (unsigned int label : g1.label) right_labels.insert(label);
    std::set<unsigned int> labels;  // labels that appear in both graphs
    std::set_intersection(std::begin(left_labels),
                          std::end(left_labels),
                          std::begin(right_labels),
                          std::end(right_labels),
                          std::inserter(labels, std::begin(labels)));

    // Create a bidomain for each label that appears in both graphs
    for (unsigned int label : labels) {
        int start_l = left.size();
        int start_r = right.size();

        for (int i=0; i<g0.n; i++)
            if (g0.label[i]==label)
                left.push_back(i);
        for (int i=0; i<g1.n; i++)
            if (g1.label[i]==label)
                right.push_back(i);

        int left_len = left.size() - start_l;
        int right_len = right.size() - start_r;
        domains.push_back({start_l, start_r, left_len, right_len, false});
    }

    vector<VtxPair> incumbent;

    if (arguments.big_first) {
        for (int k=0; k<g0.n; k++) {
            unsigned int goal = g0.n - k;
            auto left_copy = left;
            auto right_copy = right;
            auto domains_copy = domains;
            vector<VtxPair> current;
            solve(g0, g1,lgrade,rgrade, incumbent, current, domains_copy, left_copy, right_copy, goal);
            if (incumbent.size() == goal || abort_due_to_timeout) break;
            if (!arguments.quiet) cout << "Upper bound: " << goal-1 << std::endl;
        }

    } else {
        vector<VtxPair> current;
        solve(g0, g1,lgrade,rgrade, incumbent, current, domains, left, right, 1);
    }

    return incumbent;
}

vector<int> calculate_degrees(const Graph & g) {
    vector<int> degree(g.n, 0);
    for (int v=0; v<g.n; v++) {
        for (int w=0; w<g.n; w++) {
            unsigned int mask = 0xFFFFu;
            if (g.adjmat[v][w] & mask) degree[v]++;
            if (g.adjmat[v][w] & ~mask) degree[v]++;  // inward edge, in directed case
        }
    }
    return degree;
}

int sum(const vector<int> & vec) {
    return std::accumulate(std::begin(vec), std::end(vec), 0);
}

int main(int argc, char** argv) {
    set_default_arguments();
    argp_parse(&argp, argc, argv, 0, 0, 0);

    char format = arguments.dimacs ? 'D' : arguments.lad ? 'L' : 'B';
    struct Graph g0 = readGraph(arguments.filename1, format, arguments.directed,
            arguments.edge_labelled, arguments.vertex_labelled);
    struct Graph g1 = readGraph(arguments.filename2, format, arguments.directed,
            arguments.edge_labelled, arguments.vertex_labelled);

  //  std::thread timeout_thread;
  //  std::mutex timeout_mutex;
    std::condition_variable timeout_cv;
    abort_due_to_timeout.store(false);
    bool aborted = false;
#if 0
    if (0 != arguments.timeout) {
        timeout_thread = std::thread([&] {
                auto abort_time = std::chrono::steady_clock::now() + std::chrono::seconds(arguments.timeout);
                {
                    /* Sleep until either we've reached the time limit,
                     * or we've finished all the work. */
                    std::unique_lock<std::mutex> guard(timeout_mutex);
                    while (! abort_due_to_timeout.load()) {
                        if (std::cv_status::timeout == timeout_cv.wait_until(guard, abort_time)) {
                            /* We've woken up, and it's due to a timeout. */
                            aborted = true;
                            break;
                        }
                    }
                }
                abort_due_to_timeout.store(true);
                });
    }
#endif
  //  auto start = std::chrono::steady_clock::now();
    clock_t start=clock();

    vector<int> g0_deg = calculate_degrees(g0);
    vector<int> g1_deg = calculate_degrees(g1);

    // As implemented here, g1_dense and g0_dense are false for all instances
    // in the Experimental Evaluation section of the paper.  Thus,
    // we always sort the vertices in descending order of degree (or total degree,
    // in the case of directed graphs.  Improvements could be made here: it would
    // be nice if the program explored exactly the same search tree if both
    // input graphs were complemented.
  #ifdef VSCORE
    vector<int> lgrade(g0.n,0);
    vector<int> rgrade(g1.n,0);
  #endif
    vector<int> vv0(g0.n);
    std::iota(std::begin(vv0), std::end(vv0), 0);
    bool g1_dense = sum(g1_deg) > g1.n*(g1.n-1);
    std::stable_sort(std::begin(vv0), std::end(vv0), [&](int a, int b) {
        return g1_dense ? (g0_deg[a]<g0_deg[b]) : (g0_deg[a]>g0_deg[b]);
    });

    vector<int> vv1(g1.n);
    std::iota(std::begin(vv1), std::end(vv1), 0);
    bool g0_dense = sum(g0_deg) > g0.n*(g0.n-1);
    std::stable_sort(std::begin(vv1), std::end(vv1), [&](int a, int b) {//????????????????????????????????????????????????????
        return g0_dense ? (g1_deg[a]<g1_deg[b]) : (g1_deg[a]>g1_deg[b]);
    });

#if 0
    int idx;
    for(idx=0;idx<vv0.size();idx++)
       cout<<vv0[idx]<<"  ";
    cout<<endl;
    for(idx=0;idx<g0_deg.size();idx++)
       cout<<g0_deg[idx]<<"  ";
    cout<<endl;
    for(idx=0;idx<vv1.size();idx++)
       cout<<vv1[idx]<<"  ";
    cout<<endl;
    for(idx=0;idx<g1_deg.size();idx++)
       cout<<"("<<idx<<","<<g1_deg[idx]<<")  ";
    cout<<endl;

#endif
    struct Graph g0_sorted = induced_subgraph(g0, vv0);
    struct Graph g1_sorted = induced_subgraph(g1, vv1);
#if 0
    int idx;
    for(idx=0;idx<g0.n;idx++)
        cout<<g0.label[idx]<<" ";
    cout<<endl;
    for(idx=0;idx<g0_sorted.n;idx++)
        cout<<g0_sorted.label[idx]<<" ";
    cout<<endl;
#endif

    vector<VtxPair> solution = mcs(g0_sorted, g1_sorted,lgrade,rgrade);

    // Convert to indices from original, unsorted graphs
    for (auto& vtx_pair : solution) {
        vtx_pair.v = vv0[vtx_pair.v];
        vtx_pair.w = vv1[vtx_pair.w];
    }

   // auto stop = std::chrono::steady_clock::now();
   // auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    clock_t time_elapsed=clock( )-start;
    clock_t time_find=bestfind - start;
    /* Clean up the timeout thread */
   #if 0
    if (timeout_thread.joinable()) {
        {
            std::unique_lock<std::mutex> guard(timeout_mutex);
            abort_due_to_timeout.store(true);
            timeout_cv.notify_all();
        }
        timeout_thread.join();
    }
#endif
    if (!check_sol(g0, g1, solution))
        fail("*** Error: Invalid solution\n");

    cout << "Solution size " << solution.size() << std::endl;
    for (int i=0; i<g0.n; i++)
        for (unsigned int j=0; j<solution.size(); j++)
            if (solution[j].v == i)
                cout << "(" << solution[j].v << " -> " << solution[j].w << ") ";
    cout << std::endl;

    cout<<"Nodes:                      " << nodes << endl;
    cout<<"Cut branches:               "<<cutbranches<<endl;
    cout<<"Conflicts:                    " <<conflicts<< endl;
    printf("CPU time (ms):              %15ld\n", time_elapsed * 1000 / CLOCKS_PER_SEC);
    printf("FindBest time (ms):              %15ld\n", time_find * 1000 / CLOCKS_PER_SEC);
  #ifdef Best
    cout<<"Best nodes:                 "<<bestnodes<<endl;
    cout<<"Best count:                 "<<bestcount<<endl;
#endif
    if (aborted)
        cout << "TIMEOUT" << endl;
}

