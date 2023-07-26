import os
import torch
from flask import request, make_response, Response, jsonify
from flask import Flask
from transformers import BertTokenizer
from py2neo import Graph
from werkzeug.utils import secure_filename
from data_util import open_json, data2sentence, pro_data2dir

"""
    Python Flask应用的初始化部分
"""
# 创建一个Flask应用对象
app = Flask(__name__)
# 设置上传文件的保存路径为`uploads/`文件夹
app.config['UPLOAD_FOLDER'] = 'uploads/'
# 设置上传文件的最大大小为16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# 创建一个Neo4j图数据库的连接对象，该连接对象使用Bolt协议连接到Neo4j数据库，指定了数据库的地址和认证信息。
graph = Graph("bolt://192.168.78.128:7687", auth=("neo4j", "neo4j123456"))

@app.route('/v1/entity_annotation/en', methods=["POST"])
def entity_annotation():
    """
    前端将输入文本传给后台，后台首先将输入文本按
    句拆分，逐句带入实体识别模型进行知识实体的标注，并将每句文本中的知识实体0
    及实体类型返回至前端
    :return:
    """

    # 获取数据集dataset
    dataset = request.values.get('dataset')

    # 如果dataset为空，则返回一个自定义的错误响应，提示输入不能为空。
    if dataset is None:
        response = make_response("输入不能为空")
        response.status_code = 550  # 响应状态码为550
        response.content_type = 'application/json'
        return response

    # 定义一个bio数据列表
    bio_data = []

    # 调用自定义‘open_json()’函数，将id2lable.json文件解析为python对象（解析后的对象可以
    # 是字典、列表、字符串、数字等等，具体取决于JSON字符串的结构）
    index_2_label = open_json('./data/id2label.json')

    # 从预训练模型中加载一个BERT分词器，并创建了一个对应的分词器对象`tokenizers`
    tokenizers = BertTokenizer.from_pretrained(os.path.join("..", "matbert-bandgap"))

    # 使用`torch.load()`函数加载了一个名为`BruceBertCrf_chang.pt`的模型文件
    model = torch.load(r"D:\py\KG\api\model\BruceBertCrf_chang.pt")

    # 调用自定义函数‘data2sentence()’函数，将数据集dataset分割成句子列表
    sentences = data2sentence(dataset)
    for sentence in sentences:
        # 将句子分割成单词列表
        text = sentence.split(" ")
        # 使用BERT分词器`tokenizers`对列表`text`进行编码
        text_idx = tokenizers.encode(text, add_special_tokens=True, return_tensors='pt')
        # 对编码后的文本`text_idx`进行前向传播
        pre = model.forward(text_idx)
        # 对变量`pre`进行了切片操作，去掉了第一个和最后一个元素
        pre = pre[0][1: -1]
        # 将变量`pre`中的索引值转换为对应的标签
        pre = [index_2_label[i] for i in pre]
        # 将经过处理的文本和对应的标签合并到一个名为`bio_data`的列表中。
        bio_data.extend([f"{w} {t}" for w, t in zip(text, pre)])
        bio_data.append('')
    response = make_response(pro_data2dir(bio_data))
    # 设置响应的状态码为200
    response.status_code = 200
    # 设置响应的数据类型为json
    response.content_type = 'application/json'
    return response


@app.route('/v1/kn_query_by_type/', methods=["POST"])
def kn_query_by_type():
    """
    用户输入某一类别的知识实体，便可在后台图数据库中转化成相应的查询语句，并
    将查询结果返回至前端进行展示，得到文献中与该知识实体有关联的其他类别的
    实体。
    :return:
    """
    # 从请求中获取参数`n`和`m`，如果参数不存在，则默认设置为1
    n = request.values.get("n")
    m = request.values.get("m")
    nodes = []
    edges = []
    if n is None:
        n = 1
    if m is None:
        m = 1

    # 从请求中获取参数`value`和`type`，表示用户输入的知识实体的值和类别
    value = request.values.get("value")
    type_ = request.values.get("type")
    print(value, type_)

    # 构建Cypher查询语句，根据输入的知识实体值和类别，在图数据库中查询与之相关的实体
    cypher = "MATCH (x:%s)-[r*%d..%d]-(y) WHERE x.name = '%s' RETURN x,r,y" % (
        type_, int(n), int(m), value)
    results = graph.run(cypher)
    # 将结果转为前端所需要的形式
    for result in results:
        x = result["x"]
        r = result["r"][0]
        y = result["y"]
        source = {'label': list(x.labels)[0], 'name': x['name']}
        target = {'label': list(y.labels)[0], 'name': y['name']}
        edge = {'source': r.start_node['name'], 'target': r.end_node['name'], 'label': type(r).__name__,
                'from': r['from']}

        # 将节点和关系添加到相应的列表中
        if source not in nodes:
            nodes.append(source)
        if target not in nodes:
            nodes.append(target)
        if edge not in edges:
            edges.append(edge)
    data = {'nodes': nodes, 'edges': edges}
    # 将字典对象`data`转换为JSON格式的响应数据
    g = jsonify(data)
    response = make_response(g)
    # 设置响应的状态码为200
    response.status_code = 200
    # 设置响应的数据类型为json
    response.content_type = 'application/json'
    return response


@app.route('/v1/get_value_by_type', methods=["GET"])
def get_value_by_type():
    """
        根据给定的类型获取相应类型的知识实体值
    """
    type_ = ["DATA", "APL", "MAT", "MET", "ATTR", "CON", "DSC"]
    # type_ = request.values.get("type")
    values = []
    for t in type_:
        cypher = "MATCH (x:%s) return x" % t
        results = graph.run(cypher)
        nodes = []
        for result in results:
            name = result['x']['name']
            if name not in nodes:
                nodes.append(name)
        values.append(nodes)
    g = jsonify(values)
    response = make_response(g)
    # 设置响应的状态码为200
    response.status_code = 200
    # 设置响应的数据类型为json
    response.content_type = 'application/json'
    return response


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}  # 允许的文件扩展名

"""
    判断文件名是否包含`.`符号，并将文件名的扩展名转换为小写形式，
    然后检查扩展名是否在允许的扩展名集合`ALLOWED_EXTENSIONS`中。
    如果满足这两个条件，函数会返回`True`，表示文件具有允许的扩
    展名；否则，函数会返回`False`，表示文件不具有允许的扩展名
   """
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
    接收前端上传的文件，如果文件不为空且符合上传类型的文件，
    则将文件保存至该'./data/'路径上
"""
@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        file.save('./data/')

    else:
        return '不允许上传该类型的文件'

"""
    用于构建知识图谱
"""
@app.route('/build_kg', methods=['POST'])
def build_kg():
    if request.method == 'POST':
        file = request.files['file']
        dst = os.path.join(os.path.dirname(__file__), file.filename)
        file.save(dst)
        with open(dst, 'r') as f:
            # 读取bio里面的文件内容
            content = f.read().split("\n")

        os.remove(dst)  # 可选，删除临时文件
        if file:
            filename = secure_filename(file.filename)  # 防止恶意文件名
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'File uploaded successfully'

"""
    它接受一个GET请求，从请求中获取要导出的数据集，
    并将数据集作为纯文本文件进行导出。导出的文件会
    以附件形式下载，文件名为`data.txt`。
"""
@app.route('/export_txt')
def export_txt():
    data = request.values.get('dataset')
    response = Response(data)
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Content-Disposition'] = 'attachment; filename=data.txt'
    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)
