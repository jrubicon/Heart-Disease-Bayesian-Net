def dataImport():
    import pandas as pd
    import numpy as np
    return pd.read_csv("heart.csv")


def modelcreator(df):
    from pgmpy.models import BayesianModel
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    #model = BayesianModel([('target','cp'),('target','thalach'),('target','slope'),('target','restecg'),
    #('cp','thalach'),('cp','slope'),('cp','restecg'),('thalach','slope'),('thalach','restecg'),('slope','restecg')])
    model = BayesianModel([('target','cp'),('target','thalach'),('target','slope'),('target','restecg')])
    model.fit(df, estimator=BayesianEstimator)
    return model


def logo():
    print("\nHEART DISEASE BAYESIAN NETWORK")
    print("Compute the probability of heart disease")
    print("Created by: Justin Drouin, Phong Nguyen, Dania Wareh\n")


def interface(model, df):
    print("\nSelect categories to determine probability of heart disease:")
    options = ["Chest Pain","Thalach","Slope","Restecgn", "RUN MODEL"]
    cont = 'y'
    querysel = []
    visited = []
    cpuniq = df.cp.unique()
    thalachuniq = df.thalach.unique()
    slopeuniq = df.slope.unique()
    restecguniq = df.restecg.unique()

    while cont == 'y' or cont == 'Y':
        for (i, item) in enumerate(options, start=1):
            if i in visited:
                continue
            print(" ",i,": ",item)
        print("")
        sel = int(input())
        if sel in visited:
            sel = 0
        if sel == 1:
            evidence = "cp"
            input("WARNING: System will error if data is not available. Check data availabilty before entering. Press Enter to continue...\n")
            print("DATA AVAILABILITY: ", df.cp.unique(),"\n")
            print("You selected chest pain. Indicate the type:\n",
            "0: no symptoms\n",
            "1: typical angina\n",
            "2: atypical angina\n",
            "3: non-anginal pain\n",
            "4: asymptomatic\n")
            cptype = int(input())
            if cptype in cpuniq:
                visited.append(1)
                querysel.append([evidence,cptype])
            else:
                print("Invalid selection")
        elif sel == 2:
            evidence = "thalach"
            input("WARNING: System will error if data is not available. Check data availabilty before entering. Press Enter to continue...\n")
            print("DATA AVAILABILITY: ", df.thalach.unique(),"\n")
            print("You selected Thalach. Indicate the maximum heart rate achieved:")
            maxim = int(input())
            if maxim in thalachuniq:
                visited.append(2)
                querysel.append([evidence,maxim])
            else:
                print("Heart Rate not found in dataset")
        elif sel == 3:
            evidence = "slope"
            input("WARNING: System will error if data is not available. Check data availabilty before entering. Press Enter to continue...\n")
            print("DATA AVAILABILITY: ", df.slope.unique(),"\n")
            print("You selected slope. Indicate the slope of the peak exercise ST segment:\n 1: upsloping\n 2: flat \n 3: downsloping\n")
            slopval = int(input())
            if slopval in slopeuniq:
                visited.append(3)
                querysel.append([evidence,slopval])
            else:
                print("Invalid selection")
        elif sel == 4:
            evidence = "restecg"
            input("WARNING: System will error if data is not available. Check data availabilty before entering. Press Enter to continue...\n")
            print("DATA AVAILABILITY: ", df.restecg.unique(),"\n")
            print("You selected restecg. Indicate the resting electrocardiographic results:\n", 
            "0: normal\n",
            "1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)\n",
            "2: showing probable or definite left ventricular hypertrophy by Estes' criteria\n")
            restval = int(input())
            if restval in restecguniq:
                visited.append(4)
                querysel.append([evidence,restval])
            else:
                print("Invalid selection")
        elif sel == 5:
            print("RUNNING PROBABILITY:")
            query(model, querysel)
            return
        else:
            print("Not a valid selection.")
        print("Continue? (y/n) Select (n) to run model")
        cont = input()
        if not(cont == 'y' or cont == "Y"):
            print("RUNNING PROBABILITY:")
            query(model, querysel)
    

def query(model, vals):
    #Query
    from pgmpy.inference import VariableElimination
    HeartDisease_infer = VariableElimination(model)
    print(vals,"\n")
    size = len(vals)
    if size == 1:
        print(HeartDisease_infer.map_query(['target'], evidence={ vals[0][0]: vals[0][1] }), "\n")
        print(HeartDisease_infer.query(['target'], evidence={ vals[0][0]: vals[0][1] }))
    elif size == 2:
        print(HeartDisease_infer.map_query(['target'], evidence={ vals[0][0]: vals[0][1], vals[1][0]: vals[1][1] }), "\n")
        print(HeartDisease_infer.query(['target'], evidence={ vals[0][0]: vals[0][1], vals[1][0]: vals[1][1] }))
    elif size == 3:
        print(HeartDisease_infer.map_query(['target'], evidence={ vals[0][0]: vals[0][1], vals[1][0]: vals[1][1], vals[2][0]: vals[2][1] }), "\n")
        print(HeartDisease_infer.query(['target'], evidence={ vals[0][0]: vals[0][1], vals[1][0]: vals[1][1], vals[2][0]: vals[2][1] }))
    elif size == 4:
        print(HeartDisease_infer.map_query(['target'], evidence={ vals[0][0]: vals[0][1], vals[1][1]: vals[1][1], vals[2][0]: vals[2][1], vals[3][0]: vals[3][1] }), "\n")
        print(HeartDisease_infer.query(['target'], evidence={ vals[0][0]: vals[0][1], vals[1][0]: vals[1][1], vals[2][0]: vals[2][1], vals[3][0]: vals[3][1] }))
    else:
        print("ERROR")
        #print(HeartDisease_infer.map_query(['target'], evidence={'thalach': 190, 'cp': 1}))
        #print(HeartDisease_infer.query(['target'], evidence={'thalach': 202, 'cp': 1}))


def main():
    logo()
    #import data
    df = dataImport()
    #create model
    model = modelcreator(df)
    print("LOADING EDGES AND INDEPENDENCIES...\n")
    print("Model Nodes = ", model.nodes())
    print("Model Edges = ", model.edges())
    print(model.get_independencies(),"\n")

    cont = 'n'
    while cont == 'n' or cont == 'N':
        interface(model, df)
        print("Exit? (y/n)\n")
        cont = input()
    print("\nExiting...\n")


if __name__ == "__main__":
    main()