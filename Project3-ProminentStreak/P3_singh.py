import pandas as pd
import time
import math

def readfiles():    
    #read from the csv file and return a Pandas DataFrame.
    nba = pd.read_csv("1991-2004-nba.dat",  delimiter='#')
        
    #Pandas DataFrame allows you to select columns. 
    #We use column selection to split the data. 
    #We only need 2 columns in the data file: Player ID and Points.
    columns = ['ID', 'PTS']
    nba_records = nba[columns]
    
    #For each player, store the player's points in all games in an ordered list.
    #Store all players' sequences in a dictionary.
    pts = {}    
    cur_player = 'NULL'
    #The data file is already sorted by player IDs, followed by dates.
    for index, row in nba_records.iterrows():
        player, points = row
        if player != cur_player:
            cur_player = player
            pts[player] = []            
        pts[player].append(points)
    return pts

def prominent_streaks(sequences):
    lps = []
    # Get local prominent streak for all the players using linear LPS method
    for player in sequences.keys():
        lps += llps(player, sequences[player])
    # Retrieve the prominent streaks using Skyline algorithm
    return skyline(lps)

def skyline(lps):
    ps = []
    for i in range(len(lps)):
        temp_ps = [] # Store the ps to facilitate updation of ps while looping
        dominated = False
        for j in range(len(ps)):
            temp_ps.append(ps[j])
            # LPS is dominated by one of the streaks stored in ps
            if compare(lps[i], ps[j]):
                dominated = True
                # Transfer rest of the ps in temp_ps
                for k in range(j+1, len(ps)):
                    temp_ps.append(ps[k])
                break # No need to compare with other streaks
            # One of the streak stored in ps is dominated by the current LPS
            if compare(ps[j], lps[i]):
                temp_ps.remove(ps[j]) # Remove the ps as it already dominated
        # If not dominated, then store it as a prominent streak
        if not dominated:
            temp_ps.append(lps[i])
        ps = temp_ps
    return ps

def compare(s1, s2):
    # Condition for dominance
    if (s2[2]>=s1[2] and s2[3]>s1[3]) or (s2[2]>s1[2] and s2[3]>=s1[3]):
        return True
    return False

# Linear local prominent streak algorithm
def llps(player, sequence):
    potential_lps = [[player, 0, 1, sequence[0]]]
    lps = []
    for i in range(1, len(sequence)):
        curr_val = sequence[i]
        temp_potential_lps = [] # duplicate list to remove elements while looping
        add_new = True # We do not add a new potential streak when there exist a streak with min value same as curr_val
        longest = -1 # Keep record of the streak having higher min value than curr_val and having longest length
        add_new_streak = [player, i, 1 ,curr_val] # Add this new streak when all existing streaks have lower min val
        # print("\n>>>>> Iteration: ", i ,", Current Value: ", curr_val)
        for j in range(len(potential_lps)):
            min_value = potential_lps[j][3]
            # print("Evaluating potential lps: ", potential_lps[j])
            if min_value <= curr_val:
                potential_lps[j][2] += 1 # increase length of streak by 1
                temp_potential_lps.append(potential_lps[j])
                # print("Min modified: ", potential_lps[j])
                if min_value == curr_val:
                    add_new = False
            else:
                lps.append([player, potential_lps[j][1], potential_lps[j][2],potential_lps[j][3]])
                # print("Max added to lps: ", potential_lps[j])
                if potential_lps[j][2] > longest:
                    longest = potential_lps[j][2]
                    potential_lps[j][2] += 1
                    potential_lps[j][3] = curr_val
                    add_new_streak = potential_lps[j]
                    # print("Adding ", add_new_value)
        if add_new:
            temp_potential_lps.append(add_new_streak)
            # print("New potential lps: ", add_new_value)
        potential_lps = temp_potential_lps

    # Add all the pending potential lps to lps
    for i in potential_lps:
        lps.append(i)
    return lps

t0 = time.time()
sequences = readfiles()
t1 = time.time()
print("Reading the data file takes ", t1-t0, " seconds.")

t1 = time.time()
streaks = prominent_streaks(sequences)
t2 = time.time()
print("Computing prominent streaks takes ", t2-t1, " seconds.")
print(streaks)
