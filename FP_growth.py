# CSDS 435, HW3 03/09/2021
# Maryam Ghasemian (mxg708)

from itertools import chain, combinations
from collections import defaultdict, OrderedDict
from csv import reader
import pandas as pd


######
# Read data from CSV
data = pd.read_csv ('G:/My Drive/Spring 2021/CSDS435/assignments/HW3/HW3codes/transaction.csv')
print(data)
min_sup_ratio = float (input ("What is minimum support ratio? "))
min_conf = float (input ("What is minimum Confidence? "))

class Node:
    def __init__(self, itemName, frequency, parentNode):
        self.itemName = itemName
        self.count = frequency
        self.parent = parentNode
        self.children = {}
        self.next = None

    def increment(self, frequency):
        self.count += frequency

    def display(self, ind=1):
        print ('  ' * ind, self.itemName, ' ', self.count)
        for child in list (self.children.values ()):
            child.display (ind + 1)



def getFromFile(data):
    itemSetList = []
    f = []
    frequency = [dict () for x in range (len (data.values[0]) + 1)]

    for i in range (0, len (data)):
        itemSetList.append ([str (data.values[i, j]) for j in range (0, len (data.values[0]))])

    for i in itemSetList:
        for j in i:
            f.append (j)

    for i in f:
        # If item is present in dictionary, increment its count by 1
        if i in frequency[1]:
            frequency[1][i] = frequency[1][i] + 1

    print("itemsets from file: \n", itemSetList)
    return itemSetList, frequency


def constructTree(itemSetList, frequency, min_sup):
    headerTable = defaultdict (int)
    # Counting frequency and create header table
    for idx, itemSet in enumerate (itemSetList):
        for item in itemSet:
            headerTable[item] += frequency[idx]

    # Deleting items below min_sup
    headerTable = dict ((item, sup) for item, sup in headerTable.items () if sup >= min_sup)
    if (len (headerTable) == 0):
        return None, None

    # HeaderTable column [Item: [frequency, headNode]]
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]

    # Init Null head node
    fpTree = Node ('Null', 1, None)
    # Update FP tree for each cleaned and sorted itemSet
    for idx, itemSet in enumerate (itemSetList):
        itemSet = [item for item in itemSet if item in headerTable]
        itemSet.sort (key=lambda item: headerTable[item][0], reverse=True)
        # Traverse from root to leaf, update tree with given item
        currentNode = fpTree
        for item in itemSet:
            currentNode = updateTree (item, currentNode, headerTable, frequency[idx])

    return fpTree, headerTable


def updateHeaderTable(item, targetNode, headerTable):
    if (headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        # Traverse to the last node then link it to the target
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode


def updateTree(item, treeNode, headerTable, frequency):
    if item in treeNode.children:
        # If the item already exists, increment the count
        treeNode.children[item].increment (frequency)
    else:
        # Create a new branch
        newItemNode = Node (item, frequency, treeNode)
        treeNode.children[item] = newItemNode
        # Link the new branch to header table
        updateHeaderTable (item, newItemNode, headerTable)

    return treeNode.children[item]


def ascendFPtree(node, prefixPath):
    if node.parent != None:
        prefixPath.append (node.itemName)
        ascendFPtree (node.parent, prefixPath)


def findPrefixPath(basePat, headerTable):
    # First node in linked list
    treeNode = headerTable[basePat][1]
    condPats = []
    frequency = []
    while treeNode != None:
        prefixPath = []
        # From leaf node all the way to root
        ascendFPtree (treeNode, prefixPath)
        if len (prefixPath) > 1:
            # Storing the prefix path and it's corresponding count
            condPats.append (prefixPath[1:])
            frequency.append (treeNode.count)

        # Go to next node
        treeNode = treeNode.next
    return condPats, frequency


def mineTree(headerTable, min_sup, preFix, freqItemList):
    # Sort the items with frequency and create a list
    sortedItemList = [item[0] for item in sorted (list (headerTable.items ()), key=lambda p: p[1][0])]
    # Start with the lowest frequency
    for item in sortedItemList:
        # Pattern growth is achieved by the concatenation of suffix pattern with frequent patterns generated from conditional FP-tree
        newFreqSet = preFix.copy ()
        newFreqSet.add (item)
        freqItemList.append (newFreqSet)
        # Find all prefix path, constrcut conditional pattern base
        conditionalPattBase, frequency = findPrefixPath (item, headerTable)
        # Construct conditonal FP Tree with conditional pattern base
        conditionalTree, newHeaderTable = constructTree (conditionalPattBase, frequency, min_sup)
        if newHeaderTable != None:
            # Mining recursively on the tree
            mineTree (newHeaderTable, min_sup,
                      newFreqSet, freqItemList)


def powerset(s):
    return chain.from_iterable (combinations (s, r) for r in range (1, len (s)))


def getSupport(testSet, itemSetList):
    count = 0
    for itemSet in itemSetList:
        if (set (testSet).issubset (itemSet)):
            count += 1
    return count


def associationRule(freqItemSet, itemSetList, min_conf):
    rules = []
    for itemSet in freqItemSet:
        subsets = powerset (itemSet)
        itemSetSup = getSupport (itemSet, itemSetList)
        for s in subsets:
            confidence = float (itemSetSup / getSupport (s, itemSetList))
            if (confidence > min_conf):
                rules.append ([set (s), set (itemSet.difference (s)), confidence])
    return rules


def getFrequencyFromList(itemSetList):
    frequency = [1 for i in range (len (itemSetList))]
    return frequency


def fpgrowth(itemSetList, min_sup_ratio, min_conf):
    frequency = getFrequencyFromList(itemSetList)
    min_sup = len(itemSetList) * min_sup_ratio
    fpTree, headerTable = constructTree(itemSetList, frequency, min_sup)
    if(fpTree == None):
        print('No frequent item set')
    else:
        freqItems = []
        mineTree(headerTable, min_sup, set(), freqItems)
        rules = associationRule(freqItems, itemSetList, min_conf)
        return freqItems, rules

def fpgrowthFromFile(data, min_sup_ratio, min_conf):
    itemSetList, frequency = getFromFile(data)
    min_sup = len(itemSetList) * min_sup_ratio
    fpTree, headerTable = constructTree(itemSetList, frequency, min_sup)
    if(fpTree == None):
        print('No frequent item set')
    else:
        freqItems = []
        mineTree(headerTable, min_sup, set(), freqItems)
        rules = associationRule(freqItems, itemSetList, min_conf)
        return freqItems, rules

if __name__ == "__main__":
    freqItemSet, rules = fpgrowthFromFile(data, min_sup_ratio, min_conf)

    print(" frequent itemsets:\n")
    print(freqItemSet)
    print("Association Rules: \n")
    print(rules)