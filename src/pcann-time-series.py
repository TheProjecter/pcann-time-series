#!/usr/bin/env python

# Copyright 2009 Vic Fryzel <vic.fryzel@gmail.com>
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Generate and train all permutations of a network on given nodes, train these
networks, and test them.  Generate data about how each network performed. This
program is exclusively intended for the prediction of time series functions via
neural networks.

Based on bpnn.py by Neil Schemenauer <nas@arctrix.com>:
http://arctrix.com/nas/python/bpnn.py

Extended to support partial connectivity, threading, and data generation by
Vic Fryzel <vic.fryzel@gmail.com>.
"""

import datetime
import getopt
import math
import random
import sys
import threading
import time

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no, wi = None, wo = None):
        # number of input, hidden, and output nodes
        self.ni = ni #+ 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        if wi:
            self.wi = wi
        else:
            self.wi = makeMatrix(self.ni, self.nh)
            # Ensure that every network has the same chance of better initial
            # weights by seeding random to a constant
            random.seed(0)
            # set them to random vaules
            for i in range(self.ni):
                for j in range(self.nh):
                    self.wi[i][j] = rand(-0.2, 0.2)

        if wo:
            self.wo = wo
        else:
            self.wo = makeMatrix(self.nh, self.no)
            # set them to random vaules
            for j in range(self.nh):
                for k in range(self.no):
                    self.wo[j][k] = rand(-2.0, 2.0)
        
        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        #if len(inputs) != self.ni-1:
        #    raise ValueError, 'wrong number of inputs'

        # input activations
        #for i in range(self.ni-1):
        for i in range(self.ni):
            self.ai[i] = sigmoid(inputs[i])
            #self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                if self.wo[j][k] != 0.0:
                    self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                if self.wi[i][j] != 0.0:
                    self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        differences = 0
        for p in patterns:
            differences = differences + ((self.update(p[0])[0] - p[1][0]) ** 2)
        return differences / len(patterns)
            
    def weights(self):
        return self.wi + self.wo

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        retval = None
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                pass
            if i == 999:
                retval = error
        return retval

def gen_weight_sets(num_input, num_hidden, num_output):
    """Generates a set of initial weights for a neural network.
    
    Weights are initialized randomly, but the random number generator is
    seeded to zero before each iteration, ensuring the same initial weights
    for each network.

    If a weight is set to 0 by this initialization process, it is on 
    purpose, in order to disconnect two nodes.  Weights are not initialized
    to 0 if the two corresponding nodes are meant to be connected.
    """
    retval = []

    # All weights are either zero or non-zero, so we have states 0 or 1.
    # So we can achieve all networks by counting in base 2 up to the maximum
    # number of bits needed to represent every weight.
    # Example: 3 inputs, 3 hidden, 1 output
    #   That's 3 * 3 input -> hidden connections
    #   And 3 * 1 hidden -> output connections
    #   3 * 3 + 3 * 1 = 9 + 3 = 12 connections, 12 weights
    #   2 ** 12 = 4096
    #   4096 == 0b111111111111, 12 binary digits, all permutations
    num_bits = num_input * num_hidden + num_hidden * num_output
    num_networks = 2 ** num_bits
    
    c = 0
    for c in range(num_networks):
        # [2:] because string is of the form 0b1111
        # Get rid of the '0b'
        bits = str(bin(c))[2:]
        
        # Make sure we've enough bits, the number may be something like
        # bin(7), which is 0b111, and not enough bits for all nodes
        # So just "pad the network" with disconnected connections, e.g. 0s
        while len(bits) < num_bits:
            bits = '0' + bits

        num_input_to_hidden_conns = num_input * num_hidden
        num_hidden_to_output_conns = num_hidden * num_output
        # Init weight lists
        input_to_hidden = []
        for i in range(num_input):
            input_to_hidden.append([])
        hidden_to_output = []
        for i in range(num_hidden):
            hidden_to_output.append([])

        # Now, for each connection in this network, should the connection exist
        # If bit == 1, add a random non-zero weight
        # If bit == 0, add a zero weight
        for i in range(num_bits):
            bit = float(bits[i])
            weight = 0.0
            if bit == 1.0:
                weight = rand(-0.2, 0.2)
                while weight == 0.0:
                    weight = rand(-0.2, 0.2)
            # Determine which input->hidden connection to weight
            if i < num_input_to_hidden_conns:
                if num_input_to_hidden_conns > num_input:
                    node = int(math.floor(float(i) / float(num_input)))
                else:
                    node = i
                input_to_hidden[node].append(weight)
            # Determine which hidden->output connection to weight
            elif i - num_input_to_hidden_conns < num_hidden_to_output_conns:
                i = i - num_input_to_hidden_conns
                if num_hidden_to_output_conns > num_hidden:
                    node = int(math.floor(float(i) / float(num_hidden)))
                else:
                    node = i
                hidden_to_output[node].append(weight)
            else:
                break
        retval.append([input_to_hidden, hidden_to_output])
    return retval

def gen_patterns(num_input, num_output, min, max, a, freq, phase, period,
                 function):
    """Generates a training or testing data set."""

    retval = []
    step = math.pi / period
    i = min
    while i <= max:
        inputs = []
        for j in range(num_input):
            inputs.append(a * function(freq * (i + j) + phase))
        outputs = []
        for j in range(num_output):
            outputs.append(a * function(freq * (i + (num_input - 1) + j)
                                        + phase))
        retval.append([inputs, outputs])

        i += step
    return retval

def train_test_networks(num_inputs, num_hidden, num_outputs,
                        training_patterns, testing_patterns, num_threads=20):
    """Trains and tests all networks with the given numbers of nodes."""

    print 'wi,wh,num connections,train time(us),train mse,test time(us),test mse'
    weight_sets = gen_weight_sets(num_inputs, num_hidden, num_outputs)
    i = 1
    num_items = len(weight_sets)
    for w in weight_sets:
        while threading.activeCount() >= num_threads:
            time.sleep(0.25)
        sys.stderr.write('(%d/%d) Starting Iteration\n' % (i, num_items))
        NetHandler(num_inputs, num_hidden, num_outputs, w, training_patterns,
                   testing_patterns).start()
        i = i + 1

class NetHandler(threading.Thread):
    """A threaded wrapper for the generation, training, and testing of a single
    network."""

    def __init__(self, num_input, num_hidden, num_output,
                 weight_set, training_patterns, testing_patterns):
        """Initialize this handler with the given parameters and data sets."""

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.weight_set = weight_set
        self.training_patterns = training_patterns
        self.testing_patterns = testing_patterns
        threading.Thread.__init__(self)

    def generate_label(self):
        """Generate a label for this network of binary bits representing the
        connection matrix of zero and non-zero weights between nodes."""

        w = self.weight_set
        label = ''
        for i in w[0]:
            for j in i:
                if j == 0.0:
                    label += '0'
                else:
                    label += '1'
        label = label + ','
        for i in w[1]:
            for j in i:
                if j == 0.0:
                    label += '0'
                else:
                    label += '1'
        return label

    def get_num_connections(self):
        """Get the number of connections in this network."""

        w = self.weight_set
        num_connections = 0
        for i in w[0]:
            for j in i:
                if j != 0.0:
                    num_connections = num_connections + 1
        for i in w[1]:
            for j in i:
                if j != 0.0:
                    num_connections = num_connections + 1
        return num_connections

    def run(self):
        """Train and test this network, and training and testing data."""

        # w[0] is a list of input->hidden weights
        # w[1] is a list of hidden->output weights
        w = self.weight_set

        label = self.generate_label()
        num_connections = self.get_num_connections()
        output = label + ',' + str(num_connections)
        
        # Create the network we'll train and test
        n = NN(self.num_input, self.num_hidden, self.num_output, w[0], w[1])
        
        # Train the network with training_patterns
        # Measure the time it takes for this network to train
        before = datetime.datetime.now()
        train_error = n.train(self.training_patterns)
        after = datetime.datetime.now()
        d = (after - before)
        output = output + ',' + str(d) + ',' + str(train_error)
        
        # Test the network with testing_patterns
        # Measure the time it takes for this network to test
        before = datetime.datetime.now()
        test_error = n.test(self.testing_patterns)
        after = datetime.datetime.now()
        d = (after - before)
        output = output + ',' + str(d) + ',' + str(test_error)
        
        print output
        sys.stdout.flush()

def usage():
    """Display usage information to STDOUT."""

    print "pcann-time-series.py - Train and test all potential networks."
    print "                       Outputs CSV dataset to STDOUT."
    print "                       Networks must be 3 layers."
    print "Usage: pcann-time-series.py [OPTIONS]"
    print "-h|--help                  Display this help"
    print "-i|--num-input num         Set the number of input nodes"
    print "-d|--num-hidden num        Set the number of hidden nodes per layer"
    print "-o|--num-output num        Set the number of output nodes"
    print "-f|--function sin|sinh|saw The time series function to use"
    print "-w|--freq num              The frequency of the wave"
    print "-a|--amp num               The amplitude of the wave"
    print "-p|--period num            The period of the wave"
    print "--phase num                The phase of the wave"
    print "--train0 theta             The first training input"
    print "--trainf theta             The last training input"
    print "--test0 theta              The first test input"
    print "--testf theta              The last test input"
    print "-t|--threads num           The number of threads to use"

def main(argv):
    """Train and test networks based on the given command line args."""
    
    try:
        longs = ["help", "num-input=", "num-hlayers=", "num-hidden=",
                 "num-output=", "init-weight=",
                 "function=", "freq=", "amp=", "phase=", "period=",
                 "train0=", "trainf=", "test0=", "testf=", "threads="]
        opts, args = getopt.getopt(argv[1:], "hi:l:d:o:f:w:a:p:t:", longs)
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    
    # Set a default config, then modify it as needed.
    config = {'num-input': 3, 'num-hidden': 3, 'num-output': 1,
              'initweight': 0.1,
              
              'function': math.sin, 'freq': 1.0, 'amp': 1.0, 'phase': 0.0,
              'period': 2 * math.pi,
              
              'train0': -100.0, 'trainf': 100.0,
              'test0': -10000.0, 'testf': 10000.0,
              'threads': 10}
    
    for o, a in opts:
        if o in ('-h', '--help'):
            usage()
            sys.exit(0)
        elif o in ('-i', '--num-input'):
            config['num-input'] = int(a)
        elif o in ('-d', '--num-hidden'):
            config['num-hidden'] = int(a)
        elif o in ('-o', '--num-output'):
            config['num-output'] = int(a)
        elif o in ('-f', '--function'):
            if a == 'sin':
                config['function'] = math.sin
            elif a == 'sinh':
                config['function'] = math.sinh
            elif a == 'saw':
                # Since a period is required to gen this function, we will
                # defer setting it until we know the period
                pass
            else:
                print "Invalid option for -f or --function."
                usage()
                sys.exit(2)
        elif o in ('-w', '--freq'):
            config['freq'] = float(a)
        elif o in ('-a', '--amp'):
            config['amp'] = float(a)
        elif o in ('-p', '--period'):
            config['period'] = float(a)
            if config['function'] not in (math.sin, math.sinh):
                config['function'] = lambda x: math.fmod(x, config['period'])
        elif o in ('--phase'):
            config['phase'] = float(a)
        elif o in ('--train0'):
            config['train0'] = float(a)
        elif o in ('--trainf'):
            config['trainf'] = float(a)
        elif o in ('--test0'):
            config['test0'] = float(a)
        elif o in ('--testf'):
            config['testf'] = float(a)
        elif o in ('-t', '--threads'):
            config['threads'] = int(a)
    
    training_patterns = gen_patterns(config['num-input'],
                                     config['num-output'],
                                     config['train0'],
                                     config['trainf'],
                                     config['amp'],
                                     config['freq'],
                                     config['phase'],
                                     config['period'],
                                     config['function'])
    testing_patterns = gen_patterns(config['num-input'],
                                    config['num-output'],
                                    config['test0'],
                                    config['testf'],
                                    config['amp'],
                                    config['freq'],
                                    config['phase'],
                                    config['period'],
                                    config['function'])
    train_test_networks(config['num-input'], config['num-hidden'],
                        config['num-output'], training_patterns, 
                        testing_patterns, config['threads'])
        
if __name__ == '__main__':
    main(sys.argv)
