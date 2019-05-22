import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Read the respective CSVs
    tenants = pd.read_csv("tenants.csv")
    rooms = pd.read_csv("roomsFinal.csv")
    print("Done with importing datasets")

    #Data Preprocessing
    print("Data preprocessing Started")
    tenantGender = pd.get_dummies(tenants.gender)
    roomGenderPreference = pd.get_dummies(rooms.genderPreference)
    for i in range(roomGenderPreference.shape[0]):
        if (roomGenderPreference.iloc[i, 2] == 1):
            roomGenderPreference.iloc[i, 0] = 1
            roomGenderPreference.iloc[i, 1] = 1
    roomGenderPreference.drop(columns="all", inplace=True, axis=1)
    tenants['married'] = tenants['married'].astype(int)
    tenants = pd.concat([tenants, tenantGender], axis=1)
    tenants = tenants.drop('gender', axis=1)
    rooms = rooms.drop(columns=["genderPreference"], axis=1)
    rooms = pd.concat([rooms, roomGenderPreference], axis=1)

    tenantMinBudget = min(tenants.minBudget)
    tenantMaxBudget = max(tenants.maxBudget)
    roomMinBudget = min(rooms.budget)
    roomMaxBudget = max(rooms.budget)
    # print(roomMaxBudget)
    theMinBudget = tenantMinBudget if tenantMinBudget < roomMinBudget else roomMinBudget
    theMaxBudget = roomMaxBudget if tenantMaxBudget < roomMaxBudget else tenantMaxBudget
    # print(theMinBudget, theMaxBudget)
    numBy1000 = (theMaxBudget // 1000) - (theMinBudget // 1000) + 1
    # print(numBy1000)
    num = (theMinBudget // 1000)
    budgetList = []
    for i in range(numBy1000):
        budgetName = str(num + i) + "000-" + str(num + i + 1) + "000"
        budgetList.append(budgetName)
    # print(budgetList)
    ageList = list(['below21', 'above21', 'above40'])

    tenantInfo = set(tenants.columns)
    tenantRemove = set(['tenantId', 'title', 'age', 'tenantName', 'emailAddress', 'minBudget', 'maxBudget', 'startDate', 'location'])
    tenantInfo = tenantInfo - tenantRemove
    # print(tenantInfo)

    roomInfo = set(rooms.columns)
    # print(roomInfo)
    roomRemove = set(['roomId', 'roomName', 'address', 'roomCount', 'location', 'availabilityStartDate', 'budget'])
    roomInfo = roomInfo - roomRemove

    unifiedColumns = list(tenantInfo.union(roomInfo))
    # print(unifiedColumns)
    unifiedColumns = unifiedColumns + ageList + budgetList
    # print(unifiedColumns)

    all_tenants = list(tenants['tenantId'])
    all_rooms = list(rooms['roomId'])

    print("Data preprocessing is done")

    tenantUnified = np.zeros(shape=(len(all_tenants), len(unifiedColumns)))
    tenantUnified = pd.DataFrame(tenantUnified, columns=unifiedColumns)
    tenantUnified.index = all_tenants
    print("Created tenantUnified Vector")
    # print(tenantUnified.shape)
    # tenantUnified.head()

    roomUnified = np.zeros(shape=(len(all_rooms), len(unifiedColumns)))
    roomUnified = pd.DataFrame(roomUnified, columns=unifiedColumns)
    roomUnified.index = all_rooms
    print("Created roomUnified Vector")

    print("Assigning the weights to tenantUnified Vectors.....")
    for i in tenantUnified.index:
        if (tenants[tenants.tenantId == i]['F'].tolist()[0] == 1):
            tenantUnified.loc[i, 'F'] = 4
        if (tenants[tenants.tenantId == i]['M'].tolist()[0] == 1):
            tenantUnified.loc[i, 'M'] = 4
        if (tenants[tenants.tenantId == i]['married'].tolist()[0] == 1):
            tenantUnified.loc[i, 'married'] = 4
        if (tenants[tenants.tenantId == i]['parking'].tolist()[0] == 1):
            tenantUnified.loc[i, 'parking'] = 2
        if (tenants[tenants.tenantId == i]['age'].tolist()[0] <= 21):
            tenantUnified.loc[i, 'below21'] = 1
        if (tenants[tenants.tenantId == i]['age'].tolist()[0] > 21 and tenants[tenants.tenantId == i]['age'].tolist()[
            0] <= 40):
            tenantUnified.loc[i, 'above21'] = 1
        if (tenants[tenants.tenantId == i]['age'].tolist()[0] > 40):
            tenantUnified.loc[i, 'above40'] = 1
        minBudget = tenants[tenants.tenantId == i]['minBudget'].tolist()[0]
        maxBudget = tenants[tenants.tenantId == i]['maxBudget'].tolist()[0]
        minBudgetNum = minBudget // 1000
        maxBudgetNum = maxBudget // 1000
        for j in range(minBudgetNum, maxBudgetNum + 1):
            budgetRange = str(j) + "000-" + str(j + 1) + "000"
            tenantUnified.loc[i, budgetRange] = 5

    print("Assigning weights to roomUnified Vectors.....")
    for i in roomUnified.index:
        if (rooms[rooms.roomId == i]['F'].tolist()[0] == 1):
            roomUnified.loc[i, 'F'] = 4
        if (rooms[rooms.roomId == i]['M'].tolist()[0] == 1):
            roomUnified.loc[i, 'M'] = 4
        if (rooms[rooms.roomId == i]['married'].tolist()[0] == 1):
            roomUnified.loc[i, 'married'] = 4
        if (rooms[rooms.roomId == i]['parking'].tolist()[0] == 1):
            roomUnified.loc[i, 'parking'] = 2
        budget = rooms[rooms.roomId == i]['budget'].tolist()[0]
        budgetNum = budget // 1000
        budgetRange = str(budgetNum) + "000-" + str(budgetNum + 1) + "000"
        roomUnified.loc[i, budgetRange] = 5

    #This is for recommending rooms for a given
    # tid = input("Enter the TenantId:")
    # TRD = sklearn.metrics.pairwise.cosine_similarity(tenantUnified.values, roomUnified.values)
    # TRD = pd.DataFrame(TRD, columns=roomUnified.index)
    # TRD.index = tenantUnified.index
    # #tid = "tenant2"
    # tanentDetails = TRD.loc[tid, :]
    # recTenantRoom = pd.DataFrame({'distance': tanentDetails.values, 'roomId': tanentDetails.index})
    # recTenantRoom = recTenantRoom.sort_values(by=['distance'], ascending=False)
    # recTenantRoom = recTenantRoom[recTenantRoom['distance'] > 0]
    # print("Similarity between tenants and rooms are calculated for building Recommendation System")
    # print("Gathering room Info....")
    # recTenantRoom['roomName'] = recTenantRoom['roomId'].apply(lambda x: rooms['roomName'][rooms['roomId'] == x].tolist()[0])
    # recTenantRoom['budget'] = recTenantRoom['roomId'].apply(lambda x: rooms['budget'][rooms['roomId'] == x].tolist()[0])
    # recTenantRoom['parking'] = recTenantRoom['roomId'].apply(lambda x: rooms['parking'][rooms['roomId'] == x].tolist()[0])
    # recTenantRoom['availabilityStartDate'] = recTenantRoom['roomId'].apply(lambda x: rooms['availabilityStartDate'][rooms['roomId'] == x].tolist()[0])
    # recTenantRoom['location'] = recTenantRoom['roomId'].apply(lambda x: rooms['location'][rooms['roomId'] == x].tolist()[0])
    # print("All rooms that are recommended to "+tid+":")
    # print(recTenantRoom.head())
    #
    # # This is for recommending similar tenants for a given tenantId
    # TTD = sklearn.metrics.pairwise.cosine_similarity(tenantUnified.values, tenantUnified.values)
    # TTD = pd.DataFrame(TTD, columns=tenantUnified.index)
    # TTD.index = tenantUnified.index
    # tenant = TTD.loc[tid, :]
    # recTenants = pd.DataFrame({'distance': tenant.values, 'tenantId': tenant.index})
    # recTenants = recTenants.sort_values(by=['distance'], ascending=False)
    # recTenants = recTenants[recTenants['tenantId'] != tid]
    # print("Similarity between tenants and other tenants are calculated for building Recommendation System")
    # print("Gathering tenants Info....")
    # recTenants['tenantName'] = recTenants['tenantId'].apply(lambda x: tenants['tenantName'][tenants['tenantId'] == x].tolist()[0])
    # recTenants['emailAddress'] = recTenants['tenantId'].apply(lambda x: tenants['emailAddress'][tenants['tenantId'] == x].tolist()[0])
    # recTenants['age'] = recTenants['tenantId'].apply(lambda x: tenants['age'][tenants['tenantId'] == x].tolist()[0])
    # recTenants['married'] = recTenants['tenantId'].apply(lambda x: tenants['married'][tenants['tenantId'] == x].tolist()[0])
    # recTenants['Female'] = recTenants['tenantId'].apply(lambda x: tenants['F'][tenants['tenantId'] == x].tolist()[0])
    # recTenants['Male'] = recTenants['tenantId'].apply(lambda x: tenants['M'][tenants['tenantId'] == x].tolist()[0])
    # recTenants['location'] = recTenants['tenantId'].apply(lambda x: tenants['location'][tenants['tenantId'] == x].tolist()[0])
    # print("All users that are similar to " + tid + ":")
    # print(recTenants.head())
    # print("_______________________________________________________________________________________")
    # print("_______________________________________________________________________________________")
    RTD = sklearn.metrics.pairwise.cosine_similarity(roomUnified.values, tenantUnified.values)
    RTD = pd.DataFrame(RTD, columns=tenantUnified.index)
    RTD.index = roomUnified.index
    rid = input("Enter the roomId:")
    print("Similarity between rooms and tenants are calculated for building Recommendation System")
    print("Gathering tenants Info....")
    room = RTD.loc[rid, :]
    recRoomTenant = pd.DataFrame({'distance': room.values, 'tenantId': room.index})
    recRoomTenant = recRoomTenant.sort_values(by=['distance'], ascending=False)
    recRoomTenant = recRoomTenant[recRoomTenant['distance'] > 0]
    # recRoomTenant.head()
    recRoomTenant['tenantName'] = recRoomTenant['tenantId'].apply(lambda x: tenants['tenantName'][tenants['tenantId'] == x].tolist()[0])
    recRoomTenant['emailAddress'] = recRoomTenant['tenantId'].apply(lambda x: tenants['emailAddress'][tenants['tenantId'] == x].tolist()[0])
    recRoomTenant['age'] = recRoomTenant['tenantId'].apply(lambda x: tenants['age'][tenants['tenantId'] == x].tolist()[0])
    recRoomTenant['married'] = recRoomTenant['tenantId'].apply(lambda x: tenants['married'][tenants['tenantId'] == x].tolist()[0])
    recRoomTenant['Female'] = recRoomTenant['tenantId'].apply(lambda x: tenants['F'][tenants['tenantId'] == x].tolist()[0])
    recRoomTenant['Male'] = recRoomTenant['tenantId'].apply(lambda x: tenants['M'][tenants['tenantId'] == x].tolist()[0])
    recRoomTenant['location'] = recRoomTenant['tenantId'].apply(lambda x: tenants['location'][tenants['tenantId'] == x].tolist()[0])
    print(recRoomTenant.head(5))

if __name__ == "__main__":
    main()
