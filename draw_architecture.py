from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='Neural Network Architecture')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2', fontname='Microsoft YaHei')

# --- 全局样式 ---
node_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#e8f4f8'}
subgraph_style = {'style': 'rounded', 'color': '#a0a0a0'}
arrow_style = {'color': '#404040'}

dot.attr('node', **node_style, fontname='Microsoft YaHei')
dot.attr('edge', **arrow_style, fontname='Microsoft YaHei')

# --- 阶段一: 1D-CNN FeatureExtractor ---
with dot.subgraph(name='cluster_cnn') as c:
    c.attr(label='阶段一: FeatureExtractor (1D ResNet)', **subgraph_style)
    c.node('input_cnn', label='<<B>输入 (日内高频数据)</B><BR/>Shape: [B, 18, 1442]>', shape='cylinder', fillcolor='#fdecdf')
    c.node('conv1', label='<Conv1d<BR/>MaxPool1d>')
    c.node('layer1', label='<ResNet Layer 1<BR/>(多个 BasicBlock1D)>')
    c.node('layer2', label='<ResNet Layer 2<BR/>(stride=2)>')
    c.node('layer3', label='<ResNet Layer 3<BR/>(stride=2)>')
    c.node('layer4', label='<ResNet Layer 4<BR/>(stride=2)>')
    c.node('avgpool', label='<AdaptiveAvgPool1d>')
    c.node('fc_cnn', label='<Linear (FC Layer)>')
    c.node('output_cnn', label='<<B>输出: "今日行情画像"</B><BR/>Shape: [B, 1024]>', shape='cylinder', fillcolor='#e6f5d0')

    c.edge('input_cnn', 'conv1', label='Shape: [B, 64, 721]')
    c.edge('conv1', 'layer1')
    c.edge('layer1', 'layer2', label='Shape: [B, 128, 361]')
    c.edge('layer2', 'layer3', label='Shape: [B, 256, 181]')
    c.edge('layer3', 'layer4', label='Shape: [B, 512, 91]')
    c.edge('layer4', 'avgpool')
    c.edge('avgpool', 'fc_cnn', label='Shape: [B, 512]')
    c.edge('fc_cnn', 'output_cnn')

# --- 数据聚合 ---
dot.node('aggregate', label='<<B>数据聚合</B><BR/>将过去60天的"画像"堆叠>', shape='folder', fillcolor='#f0f0f0')
dot.edge('output_cnn', 'aggregate')

# --- 阶段二: StockViT (Transformer) ---
with dot.subgraph(name='cluster_vit') as c:
    c.attr(label='阶段二: StockViT (Transformer)', **subgraph_style)
    c.node('input_vit', label='<输入序列<BR/>Shape: [B, 60, 1024]>')
    c.node('cls_token', label='<拼接 [CLS] Token<BR/>+<BR/>添加 Positional Embedding>')
    c.node('transformer_blocks', label='<N x Transformer Blocks<BR/>(Multi-Head Self-Attention and MLP)>')
    c.node('extract_cls', label='<提取 [CLS] Token 的输出>')
    c.node('output_vit', label='<<B>输出: "跨周期趋势总结"</B><BR/>Shape: [B, 1024]>', shape='cylinder', fillcolor='#e6f5d0')

    c.edge('input_vit', 'cls_token', label='Shape: [B, 61, 1024]')
    c.edge('cls_token', 'transformer_blocks')
    c.edge('transformer_blocks', 'extract_cls', label='Shape: [B, 61, 1024]')
    c.edge('extract_cls', 'output_vit')

dot.edge('aggregate', 'input_vit')

# --- 阶段三: 预测头 ---
with dot.subgraph(name='cluster_heads') as c:
    c.attr(label='阶段三: 四个独立的预测头 (MLPs)', **subgraph_style)
    c.node('head_max_val', label='<预测最高价值<BR/>Shape: [B, 1]>')
    c.node('head_min_val', label='<预测最低价值<BR/>Shape: [B, 1]>')
    c.node('head_max_day', label='<预测最高点日期<BR/>Shape: [B, 60]>')
    c.node('head_min_day', label='<预测最低点日期<BR/>Shape: [B, 60]>')

dot.edge('output_vit', 'head_max_val')
dot.edge('output_vit', 'head_min_val')
dot.edge('output_vit', 'head_max_day')
dot.edge('output_vit', 'head_min_day')

# --- 最终输出 ---
dot.node('final_output', label='<<B>最终输出字典</B>>', shape='ellipse', fillcolor='#fff8e1')
dot.edge('head_max_val', 'final_output', style='dashed')
dot.edge('head_min_val', 'final_output', style='dashed')
dot.edge('head_max_day', 'final_output', style='dashed')
dot.edge('head_min_day', 'final_output', style='dashed')

# --- 保存文件 ---
try:
    dot.render('model_architecture', format='png', view=False, cleanup=True)
    print("成功！神经网络结构图已保存为 'model_architecture.png'")
except Exception as e:
    print(f"生成失败: {e}")
    print("请确保您已经正确安装了 Graphviz (系统程序和Python库)。")
