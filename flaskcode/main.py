from flask import Flask, render_template
from flask_socketio import SocketIO, emit,send
from flask_socketio import join_room, leave_room
from flask import  request,  redirect,session
from flask import jsonify,send_from_directory
import pymysql,traceback
import os
import datetime
import random
import lsh
from flask import make_response
import shutil,zipfile
from searchimg import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
 
# socketio = SocketIO()
# socketio.init_app(app, cors_allowed_origins='*')
socketio = SocketIO(app, cors_allowed_origins='*') 
name_space = '/test'

basedir = os.path.abspath(os.path.dirname(__file__))
# search_path = '/search'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def create_id(): #生成唯一的图片的名称字符串，防止图片显示时的重名问题

    # return '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()) + ''.join(
    #     [str(random.randint(1, 10)) for i in range(5)])
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f");  # 生成当前时间
    # randomNum = random.randint(0, 100);  # 生成的随机整数n，其中0<=n<=100
    # if randomNum <= 10:
    #     randomNum = str(0) + str(randomNum);
    randomNum=''
    for i in range(5):
        randomNum += str(random.randint(1, 10))
    uniqueNum = str(nowTime) + str(randomNum);
    return uniqueNum;

def storage_sql(sql):
    db = pymysql.connect(host = 'localhost',user='root',password='root123456',database='user',charset='utf8')
    cursor = db.cursor()    
    try:
        cursor.execute(sql)
        db.commit()
        # print (query)
               
    except:
        traceback.print_exc()
        db.rollback
        db.close()
        return 'error'
    db.close()

def select_sql(sql,param):
    db = pymysql.connect(host = 'localhost',user='root',password='root123456',database='user',charset='utf8')
    cursor = db.cursor() 
    
       
    try:
        cursor.execute(sql,param)
        query = cursor.fetchall()
        db.commit()
        # print (query)
        return query
    except:
        traceback.print_exc()
        db.rollback
        db.close()
        return 'error'
    db.close()

def makedir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        
def deletedir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    else:
        shutil.rmtree(dirpath)
        os.mkdir(dirpath)



def zip_file(src_dir,zip_dir):
    # deletedir(zip_dir)
    zip_name = create_id()
    zip = zipfile.ZipFile(os.path.join(zip_dir,zip_name +'.zip'),'w',zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir,topdown=False):
        fpath = dirpath.replace(src_dir,'')
        fpath = fpath and fpath + os.sep or ''
        print(fpath)
        for filename in filenames:
            zip.write(os.path.join(dirpath, filename),zip_name+'/'+fpath+filename)
            print (fpath+filename)
        
    zip.close()
    return zip_name +'.zip'




@app.route('/test', methods=["POST", "GET"])
def test():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    # elif request.method=="GET":
    #     return render_template('./search.html')
    search_path = basedir + '/storage/' + user_info +'/search'
    makedir(search_path)
    compress_path = search_path + '/zip-file'
    makedir(compress_path)
    makedir(search_path+'/encodedpic')
    zip_file(search_path+'/encodedpic',search_path+'/zip-file')
    # shutil.rmtree(search_path)
    # zip_file(search_path)
    return jsonify({"msg": "图像上传成功"})

@app.route('/result1', methods=["POST", "GET"])
def result1():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    # elif request.method=="GET":
    else :
        print("--------------------------------------------")
        # src = basedir+"/storage/"+user_info+"/2022070421124976186888.jpg"
        # /home/ubuntu/code/test/new/storage/aaa/search/zip-file/
        src = basedir+"/storage/"+user_info+'/search/zip-file'
        # image_data = open(src, "rb").read()
        # response = make_response(image_data)
        # response.headers['Content-Type'] = 'image/jpg'
        # return response
        for dirpath, dirnames, filenames in os.walk(src):
            for f in filenames:
                filename = f
            
        # filename="2022103121094535863772168.zip"
        # # return send_file(src, mimetype='image/gif')
        # return  render_template('login.html')
        return send_from_directory(src,filename,as_attachment=True)
        
@app.route('/zip/<p1>/<p2>', methods=["POST", "GET"])
def result2(p1,p2):   
    src = basedir +  "/storage/" + p1 +'/send/zip'
    return send_from_directory(src,p2,as_attachment=True)


@socketio.on('answer', namespace=name_space)
def on_send(data):
    username = session.get('user_info')
    msg = data['msg']
    # if msg=="OK":
    strlist = msg.split(" ")
    to_name = strlist[0]   
    sql_0 = "INSERT INTO authority(from_name,to_name) VALUES (%s,%s)"
    sql_1 = sql_0 % (repr(username), repr(to_name))
    storage_sql(sql_1)

    
    

@socketio.on('sendmsg', namespace=name_space)
def on_send(data):
    print('ssssssssssssssssssssssss')
    username = session.get('user_info')
    room = data['room']
    msg = data['msg']
    print(username +' says: '+msg+',,,,,,'+room)
    # send(username +' says: '+msg, room=room)
    emit('receivemsg',username +' says: '+msg, room=room)
     
@socketio.on('connect', namespace=name_space)
def connected_msg():
    print('client connected.')
    username = session.get('user_info')
    join_room(username)
    # send(username + ' has entered the room.', room=username)
 
@socketio.on('disconnect', namespace=name_space)
def disconnect_msg():
    print('client disconnected.')
    username = session.get('user_info')
    leave_room(username)
    # send(username + ' has leaved the room.', room=username)



@app.route('/', methods=["POST", "GET"])
def web():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    return redirect('/home')

@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method=='GET':
        return  render_template('login.html')
    user=request.form.get('user')
    pwd=request.form.get('pwd')
    # if user=='admin' and pwd=='123':#这里可以根据数据库里的用户和密码来判断，因为是最简单的登录界面，数据库学的不是很好，所有没用。
    #     session['user_info']=user
    #     return redirect('/index')
    db = pymysql.connect(host = 'localhost',user='root',password='root123456',database='user',charset='utf8')
    cursor = db.cursor()
    sql = "select * from person where name=%s and pwd=%s"
    param = (user,pwd)
    print (param)
    try:
        cursor.execute(sql,param)
        query = cursor.fetchone()
        print (query)
        if query:
            session['user_info']=user
            db.close()
            return  redirect('/home')
        else:
            return  render_template('login.html',msg='用户名或密码输入错误')
        
    except:
        traceback.print_exc()
        db.rollback
        db.close()
        return 'error'

@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == 'GET':
        return render_template('./register.html')
    username = request.form.get('user')
    pwd1 = request.form.get('pwd1')
    pwd2 = request.form.get('pwd2')
    email = request.form.get('email')
    # print (username)\
    
    if pwd1 != pwd2:
        return render_template('register.html',msg='两次密码输入不一致')
    
    else:
        db = pymysql.connect(host = 'localhost',user='root',password='root123456',database='user',charset='utf8')
        cursor = db.cursor()
        sql = "select * from person where name=%s"
        param = (username)
        try:
            cursor.execute(sql,param)
            query = cursor.fetchone()
            print (query)
            if query:
                return render_template('register.html',msg='用户名已被注册')
            else:
                sql_0="INSERT INTO person(name,pwd,email) VALUES (%s,%s,%s)"
                sql_1=sql_0 % (repr(username), repr(pwd1),repr(email))
                cursor.execute(sql_1)
                db.commit()
                db.close()
                folder = basedir +'/storage/'+username
                # print (folder)
                # print (os.path)
                if not os.path.exists(folder):
                    # os.makedirs(folder)#创建多层目录
                    os.mkdir(folder)
                return  redirect('/login')                
        except:
            traceback.print_exc()
            db.rollback
            db.close()
            return 'error'
        #sql = "INSERT INTO person(id,name,pwd,email) VALUES(null,"+username+","+pwd1+","+email+")"

@app.route('/home', methods=["POST", "GET"])
def home():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    my = []
    return render_template('./home.html',msg=user_info ,lists=my)

@app.route('/search', methods=["POST", "GET"])
def search():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    elif request.method=="GET":
        return render_template('./search.html')
        
    elif request.method == "POST" :
        queryimg = request.files.get('file')
        querydir = basedir + "/storage/" + user_info + '/query'
        deletedir(querydir)
        imgpath = path + 'queryimg.jpg'
        file.save(imgpath)
        list1,list2 = querylist(imgpath)
        with open('./storage/images.txt', "r") as f:
            lines =f.readlines()
        list = list1 + list2

        n = 1
        for i in list:
            path = lines[i].strip().split()[0]
            result = querydir+'/result'
            makedir(result)
            shutil.copy2(os.path.join(basedir + "/storage/",path),result+'/'+str(n)+'.jpg')
            n +=1

        return redirect('/result')
    
@app.route('/result', methods=["POST", "GET"])
def result():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    # elif request.method=="GET":
    else :
        resultpath = basedir + "/storage/" + user_info + '/query'
        zip_name = zip_file(resultpath+'/result',resultpath)
        
        return send_from_directory(resultpath,zip_name,as_attachment=True)

@app.route('/upload', methods=["POST", "GET"])
def upload():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    elif request.method=="GET":
        return render_template('./upload.html')
    elif request.method == "POST" :
        f = request.files.getlist('file')
        # f = request.files
        print(request.files)
        print (f[0].filename)
        print(basedir)
        user_info = session.get('user_info')
        # path = basedir+"/static/img/"
        path = basedir + "/storage/" + user_info + "/"
        print(path)
        for file in f:
            if not (file and allowed_file(file.filename)):
                return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        for file in f:  
            print(file.filename)
            # file_path = path + file.filename.rsplit('/',1)[1]
            rename = create_id() +"." + file.filename.rsplit('.',1)[1]
            file_path = path + rename
            file.save(file_path)
            db = pymysql.connect(host = 'localhost',user='root',password='root123456',database='user',charset='utf8')
            cursor = db.cursor()
            
            try:
                
                sql_0="INSERT INTO image(name,path,username) VALUES (%s,%s,%s)"
                sql_1=sql_0 % (repr(rename), repr(file_path),repr(user_info))
                cursor.execute(sql_1)
                db.commit()
                # db.close()
                # # print(request.files["file"])
                print("4444444444444444444444111111111111111111111111111111111111111111111111111111")
                # print(request.files["file1"])
                # print(f["file0"])

                
            except:
                traceback.print_exc()
                db.rollback
                db.close()
                return 'error'
        
        db.close()
        return jsonify({"msg": "图像上传成功"})


@app.route('/sendpic', methods=["POST", "GET"])
def sendpic():
    user_info = session.get('user_info')
    if not user_info:
        return redirect('/login')
    elif request.method == "GET":
        return render_template('./send.html')
    elif request.method == "POST" :
        print ('555555555555555555555')
        print (request.form)
        room = request.form.get('room')
        msg = request.form.get('msg')
        print (room)
        print (msg)
        
        files = request.files.getlist('file')
        
        path = basedir + "/storage/" + user_info + "/send"
        print(path) 
        makedir(path)
        save_path = path + '/pic'
        deletedir(save_path)
        zip_path = path + '/zip'
        makedir(zip_path)

        for file in files:
            if not (file and allowed_file(file.filename)):
                return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        
        for file in files:  
            print(file.filename)
            name = file.filename.rsplit('/',1)[1]
            print(name)
            file.save(os.path.join(save_path,name))
            
        zip_name = zip_file(save_path,zip_path)
        print('//////////////////')
        print(zip_name)
            # file_path = path + file.filename.rsplit('/',1)[1]
            # rename = create_id() +"." + file.filename.rsplit('.',1)[1]
            # file_path = path + rename
            # file.save(file_path)
#             datetime.datetime.now().strftime("%Y-%d-%m %H-%M-%S")
# '2020-12-03 10-31-47'
        msg1 = user_info +' says: '+msg +';'+'<a href="./zip/'+user_info+'/'+zip_name+'"><点此下载></a>'
        # emit('receivemsg',msg1, room=room, namespace=name_space)
        emit('receivepic',{'a':msg1,'b':zip_name}, room=room, namespace=name_space)    
        return jsonify({"msg": "图像上传成功"})


@app.route('/authority', methods=["POST", "GET"])
def authority():
    user_info=session.get('user_info')
    if not user_info:
        return redirect('/login')
    sql = "select to_name from authority where from_name=%s"
    param = (user_info)
    list1 = select_sql(sql,param)
    print (list1)
    print (type(list1))
    return render_template('./authority1.html',list1=list1)

@app.route('/quit', methods=["POST", "GET"])
def quit():
    del session['user_info']
    return redirect('/login')


    




# def net_main(aport):
#     global app

#     #CORS(app, supports_credentials=True)   #python支持跨域  https://www.jianshu.com/p/26b561a10bec
#     #CORS(app, resources=r'/*')
#     app.run(host="0.0.0.0", port=aport, threaded=True, debug=True,  use_reloader=False ) #, debug=True




if __name__ == '__main__':
 
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)




    # 开启多线程,多线程和多进程只能开启一个,否则会出错

