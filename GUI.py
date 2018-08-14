#libraries
import xlsxwriter
from tkinter import *
from tkinter import ttk
from tkinter import simpledialog as sd
from tkinter import messagebox as mb
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import time
from collections import Counter as ct
from tkinter.messagebox import showinfo as si
from tkinter.messagebox import showwarning as sw
import sys


#import matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

#machine learning libraries : sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#warning function
def warning_f():
    sw("Caution!!!","Don't start the program without an input file, it will lead an Error!!!") 

#exit function
def program_quit():
    root.destroy()

#help function
def help_f():
    si("HELP","1.Start this program by select Execute on menus. \n2.Enter your file name (e.g = Test.csv). \n3.The naive bayes output file will be printed in xlsx \n4.Choose yes if you want to show the Pie Chart. \nNotes : \n- Output file name = \'Output.xlsx\' \n- File location is in the same folder of the program")

#Program Main Function
def func_p():
    
    file_name = sd.askstring("File Name","Enter your file name ")
    
    #data
    data = pd.read_csv(file_name)

    window = Toplevel(root)
    window.title("Function")
    window.geometry("600x500+0+0")
    window.resizable(False,False)
    window.configure(background='White')
    #Header
    window.style = ttk.Style()
    window.style.configure('TFrame', background = 'white')
    window.style.configure('TLabel', background = 'white')
    window.style.configure('Header.TLabel', font = ('Consolas',10,'bold'))
    window.style.configure('Content.TLabel', font = ('Consolas',8,'bold'))
    window.frame_header = ttk.Frame(window)
    window.frame_header.pack()
    window.logo = PhotoImage(file = '2.png')
    ttk.Label(window.frame_header, image = window.logo).grid(row = 0, column = 0, rowspan = 2,sticky = 'w',padx = 1)
    ttk.Label(window.frame_header, text = "Your file name : " + file_name,style = 'Header.TLabel').grid(row = 0, column = 1, padx = 5)
    ttk.Label(window.frame_header, text = "Note :\n1] Data Parameter Name      : Reference Data. \n2] \'Positive\' Data Parameter : Symbol that shows if the Patient has the Disease. \n3] Number of Parameter      : Number of Data Variable Parameter.\n").grid(row = 1, column = 1,sticky = 'w')
    #content
    ttk.Label(window.frame_header, text = "Data Parameter Name ").grid(row = 3, column = 0, padx = 5)
    base_data1 = StringVar()
    entry_box1 = Entry(window.frame_header,textvariable=base_data1,width=30,bg="Light Grey").grid(row = 3,column = 1)
    ttk.Label(window.frame_header, text = "\'Positive\' Data Notation ").grid(row = 4, column = 0, padx = 5)
    base_data0 = StringVar()
    entry_box0 = Entry(window.frame_header,textvariable=base_data0,width=30,bg="Light Grey").grid(row = 4,column = 1)
    ttk.Label(window.frame_header, text = "Number of Data Variable").grid(row = 5, column = 0)
    num_data1 = IntVar()
    entry_boxnum = Entry(window.frame_header,textvariable=num_data1,width=30,bg="Light Grey").grid(row = 5,column = 1)

    def do_it():

        #data parameter and base true data
        base_par = str(base_data1.get())
        base_data = str(base_data0.get())
        #Number of Variable
        i = int(num_data1.get())
        
        if base_data == "1" or base_data == "0" :
            basedata = int(base_data)
        else :
            basedata = str(base_data)

        datax = data
        datax["Patient Condition"] = np.where(data[base_par]==basedata,"Positve","Negative")

        #Mapping Data
        if basedata is not None :
            ttk.Label(window.frame_header, text = "Data Parameter Variable",style = 'Content.TLabel').grid(row = 7, column = 0)

            #loop input
            if i == 1 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)

                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable_array = [variable1]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=10, column = 1)
            elif i == 2 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)

                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable_array = [variable1,variable2]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=10, column = 1)

                
                    
                
            elif i == 3 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)

                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable_array = [variable1,variable2,variable3]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=11, column = 1)
            elif i==4 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)

                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable_array = [variable1,variable2,variable3,variable4]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=12, column = 1)
            elif i == 5 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                #entry number 5
                ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                data_var5 = StringVar()
                entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)

                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable5 = str(data_var5.get())
                    variable_array = [variable1,variable2,variable3,variable4,variable5]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=13, column = 1)
            elif i == 6 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                #entry number 5
                ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                data_var5 = StringVar()
                entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                #entry number 6
                ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                data_var6 = StringVar()
                entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)

                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable5 = str(data_var5.get())
                    variable6 = str(data_var6.get())
                    variable_array = [variable1,variable2,variable3,variable4,variable5,variable6]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=14, column = 1)
            elif i == 7 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                #entry number 5
                ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                data_var5 = StringVar()
                entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                #entry number 6
                ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                data_var6 = StringVar()
                entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)
                #entry number 7
                ttk.Label(window.frame_header, text = "Data Variable number 7").grid(row = 14, column = 0)
                data_var7 = StringVar()
                entry_var7 = Entry(window.frame_header,textvariable=data_var7,width=30,bg="Light Grey").grid(row = 14,column = 1)

                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable5 = str(data_var5.get())
                    variable6 = str(data_var6.get())
                    variable7 = str(data_var7.get())
                    variable_array = [variable1,variable2,variable3,variable4,variable5,variable6,variable7]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=15, column = 1)
            elif i == 8 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                #entry number 5
                ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                data_var5 = StringVar()
                entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                #entry number 6
                ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                data_var6 = StringVar()
                entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)
                #entry number 7
                ttk.Label(window.frame_header, text = "Data Variable number 7").grid(row = 14, column = 0)
                data_var7 = StringVar()
                entry_var7 = Entry(window.frame_header,textvariable=data_var7,width=30,bg="Light Grey").grid(row = 14,column = 1)
                #entry number 8
                ttk.Label(window.frame_header, text = "Data Variable number 8").grid(row = 15, column = 0)
                data_var8 = StringVar()
                entry_var8 = Entry(window.frame_header,textvariable=data_var8,width=30,bg="Light Grey").grid(row = 15,column = 1)
                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable5 = str(data_var5.get())
                    variable6 = str(data_var6.get())
                    variable7 = str(data_var7.get())
                    variable7 = str(data_var8.get())
                    variable_array = [variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=16, column = 1)
            elif i == 9 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                #entry number 5
                ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                data_var5 = StringVar()
                entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                #entry number 6
                ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                data_var6 = StringVar()
                entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)
                #entry number 7
                ttk.Label(window.frame_header, text = "Data Variable number 7").grid(row = 14, column = 0)
                data_var7 = StringVar()
                entry_var7 = Entry(window.frame_header,textvariable=data_var7,width=30,bg="Light Grey").grid(row = 14,column = 1)
                #entry number 8
                ttk.Label(window.frame_header, text = "Data Variable number 8").grid(row = 15, column = 0)
                data_var8 = StringVar()
                entry_var8 = Entry(window.frame_header,textvariable=data_var8,width=30,bg="Light Grey").grid(row = 15,column = 1)
                #entry number 9
                ttk.Label(window.frame_header, text = "Data Variable number 9").grid(row = 16, column = 0)
                data_var9 = StringVar()
                entry_var9 = Entry(window.frame_header,textvariable=data_var9,width=30,bg="Light Grey").grid(row = 16,column = 1)
                
                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable5 = str(data_var5.get())
                    variable6 = str(data_var6.get())
                    variable7 = str(data_var7.get())
                    variable8 = str(data_var8.get())
                    variable9 = str(data_var9.get())
                    variable_array = [variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,variable9]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=17, column = 1)
            elif i == 10 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                #entry number 5
                ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                data_var5 = StringVar()
                entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                #entry number 6
                ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                data_var6 = StringVar()
                entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)
                #entry number 7
                ttk.Label(window.frame_header, text = "Data Variable number 7").grid(row = 14, column = 0)
                data_var7 = StringVar()
                entry_var7 = Entry(window.frame_header,textvariable=data_var7,width=30,bg="Light Grey").grid(row = 14,column = 1)
                #entry number 8
                ttk.Label(window.frame_header, text = "Data Variable number 8").grid(row = 15, column = 0)
                data_var8 = StringVar()
                entry_var8 = Entry(window.frame_header,textvariable=data_var8,width=30,bg="Light Grey").grid(row = 15,column = 1)
                #entry number 9
                ttk.Label(window.frame_header, text = "Data Variable number 9").grid(row = 16, column = 0)
                data_var9 = StringVar()
                entry_var9 = Entry(window.frame_header,textvariable=data_var9,width=30,bg="Light Grey").grid(row = 16,column = 1)
                #entry number 10
                ttk.Label(window.frame_header, text = "Data Variable number 10").grid(row = 17, column = 0)
                data_var10 = StringVar()
                entry_var10 = Entry(window.frame_header,textvariable=data_var10,width=30,bg="Light Grey").grid(row = 17,column = 1)
                
                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable5 = str(data_var5.get())
                    variable6 = str(data_var6.get())
                    variable7 = str(data_var7.get())
                    variable8 = str(data_var8.get())
                    variable9 = str(data_var9.get())
                    variable10 = str(data_var10.get())
                    variable_array = [variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,variable9,variable10]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=18, column = 1)
                
            elif i == 11 :
                    #entry number 1
                    ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                    data_var1 = StringVar()
                    entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                    #entry number 2
                    ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                    data_var2 = StringVar()
                    entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                    #entry number 3
                    ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                    data_var3 = StringVar()
                    entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                    #entry number 4
                    ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                    data_var4 = StringVar()
                    entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                    #entry number 5
                    ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                    data_var5 = StringVar()
                    entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                    #entry number 6
                    ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                    data_var6 = StringVar()
                    entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)
                    #entry number 7
                    ttk.Label(window.frame_header, text = "Data Variable number 7").grid(row = 14, column = 0)
                    data_var7 = StringVar()
                    entry_var7 = Entry(window.frame_header,textvariable=data_var7,width=30,bg="Light Grey").grid(row = 14,column = 1)
                    #entry number 8
                    ttk.Label(window.frame_header, text = "Data Variable number 8").grid(row = 15, column = 0)
                    data_var8 = StringVar()
                    entry_var8 = Entry(window.frame_header,textvariable=data_var8,width=30,bg="Light Grey").grid(row = 15,column = 1)
                    #entry number 9
                    ttk.Label(window.frame_header, text = "Data Variable number 9").grid(row = 16, column = 0)
                    data_var9 = StringVar()
                    entry_var9 = Entry(window.frame_header,textvariable=data_var9,width=30,bg="Light Grey").grid(row = 16,column = 1)
                    #entry number 10
                    ttk.Label(window.frame_header, text = "Data Variable number 10").grid(row = 17, column = 0)
                    data_var10 = StringVar()
                    entry_var10 = Entry(window.frame_header,textvariable=data_var10,width=30,bg="Light Grey").grid(row = 17,column = 1)
                    #entry number 11
                    ttk.Label(window.frame_header, text = "Data Variable number 11").grid(row = 18, column = 0)
                    data_var11 = StringVar()
                    entry_var11 = Entry(window.frame_header,textvariable=data_var11,width=30,bg="Light Grey").grid(row = 18,column = 1)
                    
                    def doit2() :
                        dataz = datax
                        variable1 = str(data_var1.get())
                        variable2 = str(data_var2.get())
                        variable3 = str(data_var3.get())
                        variable4 = str(data_var4.get())
                        variable5 = str(data_var5.get())
                        variable6 = str(data_var6.get())
                        variable7 = str(data_var7.get())
                        variable8 = str(data_var8.get())
                        variable9 = str(data_var9.get())
                        variable10 = str(data_var10.get())
                        variable11 = str(data_var11.get())
                        variable_array = [variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,variable9,variable10,variable11]
                        variable_array.append ("Patient Condition")
                        dataz = dataz[variable_array].dropna(axis=0,how='any')

                        #split dataset
                        train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                        gnb = GaussianNB()
                        newarr = []
                        newarr.extend(variable_array)
                        newarr.remove("Patient Condition")

                        gnb.fit(train[newarr].values, train["Patient Condition"])
                        result = gnb.predict(test[newarr])

                        # Print Performance Indicator
                        data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                              .format(
                                  test.shape[0],
                                  (test["Patient Condition"] != result).sum(),
                                  100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                                  ))

                        test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                        test_data["Patient Condition"] = result
                        test_data["Data Accuracy"] = data_accuracy

                        #excel writer
                        writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                        test_data.to_excel(writer,sheet_name='Sheet1')
                        writer.save()
                        window.destroy()
                        si("","Output Created! Check it out!")

                    enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=19, column = 1)

            elif i == 12 :
                    #entry number 1
                    ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                    data_var1 = StringVar()
                    entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                    #entry number 2
                    ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                    data_var2 = StringVar()
                    entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                    #entry number 3
                    ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                    data_var3 = StringVar()
                    entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                    #entry number 4
                    ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                    data_var4 = StringVar()
                    entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                    #entry number 5
                    ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                    data_var5 = StringVar()
                    entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                    #entry number 6
                    ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                    data_var6 = StringVar()
                    entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)
                    #entry number 7
                    ttk.Label(window.frame_header, text = "Data Variable number 7").grid(row = 14, column = 0)
                    data_var7 = StringVar()
                    entry_var7 = Entry(window.frame_header,textvariable=data_var7,width=30,bg="Light Grey").grid(row = 14,column = 1)
                    #entry number 8
                    ttk.Label(window.frame_header, text = "Data Variable number 8").grid(row = 15, column = 0)
                    data_var8 = StringVar()
                    entry_var8 = Entry(window.frame_header,textvariable=data_var8,width=30,bg="Light Grey").grid(row = 15,column = 1)
                    #entry number 9
                    ttk.Label(window.frame_header, text = "Data Variable number 9").grid(row = 16, column = 0)
                    data_var9 = StringVar()
                    entry_var9 = Entry(window.frame_header,textvariable=data_var9,width=30,bg="Light Grey").grid(row = 16,column = 1)
                    #entry number 10
                    ttk.Label(window.frame_header, text = "Data Variable number 10").grid(row = 17, column = 0)
                    data_var10 = StringVar()
                    entry_var10 = Entry(window.frame_header,textvariable=data_var10,width=30,bg="Light Grey").grid(row = 17,column = 1)
                    #entry number 11
                    ttk.Label(window.frame_header, text = "Data Variable number 11").grid(row = 18, column = 0)
                    data_var11 = StringVar()
                    entry_var11 = Entry(window.frame_header,textvariable=data_var11,width=30,bg="Light Grey").grid(row = 18,column = 1)
                    #entry number 12
                    ttk.Label(window.frame_header, text = "Data Variable number 12").grid(row = 19, column = 0)
                    data_var12 = StringVar()
                    entry_var12 = Entry(window.frame_header,textvariable=data_var12,width=30,bg="Light Grey").grid(row = 19,column = 1)
                    
                    def doit2() :
                        dataz = datax
                        variable1 = str(data_var1.get())
                        variable2 = str(data_var2.get())
                        variable3 = str(data_var3.get())
                        variable4 = str(data_var4.get())
                        variable5 = str(data_var5.get())
                        variable6 = str(data_var6.get())
                        variable7 = str(data_var7.get())
                        variable8 = str(data_var8.get())
                        variable9 = str(data_var9.get())
                        variable10 = str(data_var10.get())
                        variable11 = str(data_var11.get())
                        variable12 = str(data_var12.get())
                        variable_array = [variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,variable9,variable10,variable11,variable12]
                        variable_array.append ("Patient Condition")
                        dataz = dataz[variable_array].dropna(axis=0,how='any')

                        #split dataset
                        train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                        gnb = GaussianNB()
                        newarr = []
                        newarr.extend(variable_array)
                        newarr.remove("Patient Condition")

                        gnb.fit(train[newarr].values, train["Patient Condition"])
                        result = gnb.predict(test[newarr])

                        # Print Performance Indicator
                        data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                              .format(
                                  test.shape[0],
                                  (test["Patient Condition"] != result).sum(),
                                  100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                                  ))

                        test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                        test_data["Patient Condition"] = result
                        test_data["Data Accuracy"] = data_accuracy

                        #excel writer
                        writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                        test_data.to_excel(writer,sheet_name='Sheet1')
                        writer.save()
                        window.destroy()
                        si("","Output Created! Check it out!")

                    enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=20, column = 1)

            elif i == 13 :
                #entry number 1
                ttk.Label(window.frame_header, text = "Data Variable number 1").grid(row = 8, column = 0)
                data_var1 = StringVar()
                entry_var1 = Entry(window.frame_header,textvariable=data_var1,width=30,bg="Light Grey").grid(row = 8,column = 1)
                #entry number 2
                ttk.Label(window.frame_header, text = "Data Variable number 2").grid(row = 9, column = 0)
                data_var2 = StringVar()
                entry_var2 = Entry(window.frame_header,textvariable=data_var2,width=30,bg="Light Grey").grid(row = 9,column = 1)
                #entry number 3
                ttk.Label(window.frame_header, text = "Data Variable number 3").grid(row = 10, column = 0)
                data_var3 = StringVar()
                entry_var3 = Entry(window.frame_header,textvariable=data_var3,width=30,bg="Light Grey").grid(row = 10,column = 1)
                #entry number 4
                ttk.Label(window.frame_header, text = "Data Variable number 4").grid(row = 11, column = 0)
                data_var4 = StringVar()
                entry_var4 = Entry(window.frame_header,textvariable=data_var4,width=30,bg="Light Grey").grid(row = 11,column = 1)
                #entry number 5
                ttk.Label(window.frame_header, text = "Data Variable number 5").grid(row = 12, column = 0)
                data_var5 = StringVar()
                entry_var5 = Entry(window.frame_header,textvariable=data_var5,width=30,bg="Light Grey").grid(row = 12,column = 1)
                #entry number 6
                ttk.Label(window.frame_header, text = "Data Variable number 6").grid(row = 13, column = 0)
                data_var6 = StringVar()
                entry_var6 = Entry(window.frame_header,textvariable=data_var6,width=30,bg="Light Grey").grid(row = 13,column = 1)
                #entry number 7
                ttk.Label(window.frame_header, text = "Data Variable number 7").grid(row = 14, column = 0)
                data_var7 = StringVar()
                entry_var7 = Entry(window.frame_header,textvariable=data_var7,width=30,bg="Light Grey").grid(row = 14,column = 1)
                #entry number 8
                ttk.Label(window.frame_header, text = "Data Variable number 8").grid(row = 15, column = 0)
                data_var8 = StringVar()
                entry_var8 = Entry(window.frame_header,textvariable=data_var8,width=30,bg="Light Grey").grid(row = 15,column = 1)
                #entry number 9
                ttk.Label(window.frame_header, text = "Data Variable number 9").grid(row = 16, column = 0)
                data_var9 = StringVar()
                entry_var9 = Entry(window.frame_header,textvariable=data_var9,width=30,bg="Light Grey").grid(row = 16,column = 1)
                #entry number 10
                ttk.Label(window.frame_header, text = "Data Variable number 10").grid(row = 17, column = 0)
                data_var10 = StringVar()
                entry_var10 = Entry(window.frame_header,textvariable=data_var10,width=30,bg="Light Grey").grid(row = 17,column = 1)
                #entry number 11
                ttk.Label(window.frame_header, text = "Data Variable number 11").grid(row = 18, column = 0)
                data_var11 = StringVar()
                entry_var11 = Entry(window.frame_header,textvariable=data_var11,width=30,bg="Light Grey").grid(row = 18,column = 1)
                #entry number 12
                ttk.Label(window.frame_header, text = "Data Variable number 12").grid(row = 19, column = 0)
                data_var12 = StringVar()
                entry_var12 = Entry(window.frame_header,textvariable=data_var12,width=30,bg="Light Grey").grid(row = 19,column = 1)
                #entry number 13
                ttk.Label(window.frame_header, text = "Data Variable number 13").grid(row = 20, column = 0)
                data_var13 = StringVar()
                entry_var13 = Entry(window.frame_header,textvariable=data_var13,width=30,bg="Light Grey").grid(row = 20,column = 1)
                
                def doit2() :
                    dataz = datax
                    variable1 = str(data_var1.get())
                    variable2 = str(data_var2.get())
                    variable3 = str(data_var3.get())
                    variable4 = str(data_var4.get())
                    variable5 = str(data_var5.get())
                    variable6 = str(data_var6.get())
                    variable7 = str(data_var7.get())
                    variable8 = str(data_var8.get())
                    variable9 = str(data_var9.get())
                    variable10 = str(data_var10.get())
                    variable11 = str(data_var11.get())
                    variable12 = str(data_var12.get())
                    variable13 = str(data_var13.get())
                    variable_array = [variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,variable9,variable10,variable11,variable12,variable13]
                    variable_array.append ("Patient Condition")
                    dataz = dataz[variable_array].dropna(axis=0,how='any')

                    #split dataset
                    train, test = train_test_split(dataz, test_size=0.6, random_state=int(4))
                    gnb = GaussianNB()
                    newarr = []
                    newarr.extend(variable_array)
                    newarr.remove("Patient Condition")

                    gnb.fit(train[newarr].values, train["Patient Condition"])
                    result = gnb.predict(test[newarr])

                    # Print Performance Indicator
                    data_accuracy = ("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
                          .format(
                              test.shape[0],
                              (test["Patient Condition"] != result).sum(),
                              100*(1-(test["Patient Condition"] != result).sum()/test.shape[0])
                              ))

                    test_data = pd.concat([test[newarr], test["Patient Condition"]], axis=1)
                    test_data["Patient Condition"] = result
                    test_data["Data Accuracy"] = data_accuracy

                    #excel writer
                    writer = pd.ExcelWriter('Output.xlsx', engine = 'xlsxwriter')
                    test_data.to_excel(writer,sheet_name='Sheet1')
                    writer.save()
                    window.destroy()
                    si("","Output Created! Check it out!")

                enter2 = Button(window.frame_header,text="Enter",width=25,command=doit2).grid(row=21, column = 1)

            else :
                sw("WARNING!!!","The number of variable parameter must within 1 to 13!") 
                
                
    
    work = Button(window.frame_header,text="Submit",width=25,command=do_it).grid(row=6, column = 1)
        
            
    
#Creating Main Window
root = Tk()
root.title("PyNB v2.0")
root.geometry("500x278+10+10")
root.resizable(False,False)

#adding menu
menu = Menu()
root.config(menu=menu)
menu.add_cascade(label='Caution',command = warning_f)
menu.add_cascade(label='Help', command = help_f)
menu.add_cascade(label='Execute', command = func_p)
menu.add_cascade(label='Quit',command = program_quit)

#Background Image
load = Image.open('c.jpg')
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
img.place(x=0,y=0)

root.mainloop()


