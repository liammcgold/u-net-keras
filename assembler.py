import numpy as np
import blender as blend
import math


class assembler(object):

    def __init__(self,raw,overlap,shape,function):
        self.raw=raw
        self.raw_shape = np.shape(raw)
        self.counter = 0
        self.overlap = overlap
        self.stage = 0
        self.shape = shape
        self.full = False
        self.function = function
        self.raw_block = None
        self.aff_block = None

        self.num_blocks,self.padding=self.calc_num_blocks()

        self.total_blocks=self.calc_total_blocks()

        self.aff_graph=self.generate_empty_aff_graph()

        self.overlap_increment = self.calc_overlap_inc()

        self.dict=self.on_edge()

        self.current_loc=self.get_raw_cordinates()

    def calc_total_blocks(self):
        total_blocks = self.num_blocks[0] * self.num_blocks[1] * self.num_blocks[2]
        return total_blocks

    def generate_empty_aff_graph(self):
        aff_graph = np.zeros((3,
                              int(self.raw_shape[0]+ self.padding[0]*self.shape[0]),
                              int(self.raw_shape[1]+ self.padding[1]*self.shape[1]),
                              int(self.raw_shape[2]+ self.padding[2]*self.shape[2])
                              ))
        return aff_graph

    def calc_overlap_inc(self):
        overlap_increment = [self.shape[0] - self.overlap * self.shape[0], self.shape[1] - self.overlap * self.shape[1],
                             self.shape[2] - self.overlap * self.shape[2]]
        return overlap_increment

    def feed(self):


        assert self.stage==0, "CALLED FEED DURRING APPEND STAGE"

        if(self.is_full()):
            print("NO MORE BLOCKS")
            return


        self.raw_block=self.get_block()

        self.stage=1


        return self.raw_block

    def increment(self):
        self.counter+=1
        self.dict=self.on_edge()
        self.full = self.is_full()
        self.stage=0
        self.current_loc=self.get_raw_cordinates()

    def append(self):
        assert self.stage==1, "CALLED APPEND DURING FEED STAGE"
        self.blend_block()
        self.increment()

    def get_block(self):

        # initialize with zeros:
        raw_block = np.zeros(self.shape)

        cords = self.current_loc

        slice = self.raw[cords[0, 0]:cords[0, 1], cords[1, 0]:cords[1, 1], cords[2, 0]:cords[2, 1]]

        slice_shape = np.shape(slice)
        raw_block[0:slice_shape[0], 0:slice_shape[1], 0:slice_shape[2]] = slice

        return raw_block

    def get_raw_cordinates(self):




        dict = self.on_edge()

        start_0 = 99999999999999999
        start_1 = 99999999999999999
        start_2 = 99999999999999999

        stop_0 = 99999999999999999
        stop_1 = 99999999999999999
        stop_2 = 99999999999999999



        # index notation
        dim_0 = self.counter % self.num_blocks[0]

        dim_1 = math.floor((self.counter % (self.num_blocks[0] * self.num_blocks[1])) / self.num_blocks[0])
        dim_2 = math.floor(self.counter / (self.num_blocks[0] * self.num_blocks[1]))

        # check positive edges and grab block from 0:size if contact
        if dict["+0"] == True:
            start_0 = 0
            stop_0 = self.shape[0]

        if dict["+1"] == True:
            start_1 = 0
            stop_1 = self.shape[1]

        if dict["+2"] == True:
            start_2 = 0
            stop_2 = self.shape[2]

        # check negative edges and if true stop early to prevent going over edge
        if dict["-0"] == True:
            start_0 = self.overlap_increment[0] * dim_0
            stop_0 = np.shape(self.raw)[0]

        if dict["-1"] == True:
            start_1 = self.overlap_increment[1] * dim_1
            stop_1 = np.shape(self.raw)[1]

        if dict["-2"] == True:
            start_2 = self.overlap_increment[2] * dim_2
            stop_2 = np.shape(self.raw)[2]

        # if neither are true assign index
        if (dict["+0"] == False and dict["-0"] == False):
            start_0 = self.overlap_increment[0] * dim_0
            stop_0 = self.shape[0] + self.overlap_increment[0] * dim_0

        if (dict["+1"] == False and dict["-1"] == False):
            start_1 = self.overlap_increment[1] * dim_1
            stop_1 = self.shape[1] + self.overlap_increment[1] * dim_1

        if (dict["+2"] == False and dict["-2"] == False):
            start_2 = self.overlap_increment[2] * dim_2
            stop_2 = self.shape[2] + self.overlap_increment[2] * dim_2

        # grab slice and add to temp array
        start_0 = int(start_0)
        start_1 = int(start_1)
        start_2 = int(start_2)
        stop_0 = int(stop_0)
        stop_1 = int(stop_1)
        stop_2 = int(stop_2)

        coordinates = [[start_0, stop_0], [start_1, stop_1], [start_2, stop_2]]

        coordinates = np.asarray(coordinates, dtype=np.int)

        '''
        print("COUNTER:    %i" % self.counter)
        print("NUM BLOCKS: ", self.num_blocks)
        print(dict)
        print("OVERLAP INCREMENT: ", self.overlap_increment)
        print("SLICE START:       ", [start_0, start_1, start_2])
        print("SLICE STOP:        ", [stop_0, stop_1, stop_2])
        print("DIMS:              ", [dim_0, dim_1, dim_2])

        print("COORDINATES:       ",coordinates)
        print("AFF GRAPH SIZE:    ",np.shape(self.aff_graph))
        '''

        return coordinates

    def on_edge(self):
        '''
        Takes counter and num of blocks in each dim and determines if there is contact between the edge of the volume and the current block

        axis are traversed in order 0->1->2

        first go through all 0 values then increment 1 until all 1 values have been incremented and then 2

        if dict returned is true that means there is contact

        '''

        total_blocks = self.total_blocks

        dict = {
            "+0": False,
            "-0": False,
            "+1": False,
            "-1": False,
            "+2": False,
            "-2": False
        }

        # check +0
        if (self.counter % self.num_blocks[0] == 0):
            dict["+0"] = True

        # check -0
        if ((self.counter + 1) % self.num_blocks[0] == 0):
            dict["-0"] = True

        # check +1
        if (self.counter % (self.num_blocks[0] * self.num_blocks[1]) < self.num_blocks[0]):
            dict["+1"] = True

        # check -1
        if (self.counter % (self.num_blocks[0] * self.num_blocks[1])) + 1 > (self.num_blocks[0] * (self.num_blocks[1] - 1)):
            dict["-1"] = True

        # check +2
        if (self.counter < self.num_blocks[0] * self.num_blocks[1]):
            dict["+2"] = True

        # check -2
        if (self.counter + 1 > (self.num_blocks[0] * self.num_blocks[1] * (self.num_blocks[2] - 1))):
            dict["-2"] = True

        return dict

    def generate_aff(self):
        self.aff_block=self.function(self.raw_block)

    def process(self):
        while(self.full==False):
            print("Block %i "%self.counter," of %i"%self.total_blocks)
            self.feed()
            self.aff_block=np.asarray(self.function(self.raw_block))
            self.append()
        self.crop_aff()
        return self.aff_graph

    def calc_num_blocks(self):
        '''
            calculate raw dim  size using formula:

            size  =  (raw_shape)/(block_size)  *  (1/overlap)  -  1

            round up to get actual number

            calc padding with formula:

            padding  =  block_size/remainder


            padding is rounded up

            '''

        num_blocks_raw = np.zeros(3)
        num_blocks = np.zeros(3)
        padding = np.zeros(3)
        for x in range(0, 3):
            num_blocks_raw[x] = (self.raw_shape[x] / self.shape[x]) * (1 / self.overlap) - 1
            num_blocks[x] = int(math.ceil(num_blocks_raw[x]))
            remainder = round(num_blocks_raw[x] % 1, 4)
            padding[x] = int(math.ceil(remainder))

        return num_blocks, padding

    def is_full(self):
        if (self.counter < self.total_blocks):
            return False
        else:
            return True

    def blend_axis0(self):
        #blends current slice with preivous slice on axis0

        ##Go back one overlap increment to find start      ##take start and add the shape to it
        prev_slice = self.aff_graph[:,
                     int(self.current_loc[0, 0] - self.overlap_increment[0]):int(self.current_loc[0, 0] - self.overlap_increment[0] + self.shape[0]),
                     int(self.current_loc[1, 0]):int(self.current_loc[1, 0] + self.shape[1]),
                     int(self.current_loc[2, 0]):int(self.current_loc[2, 0]+ self.shape[2])]



        blended_slice = blend.blend(prev_slice, self.aff_block, "0+",self.overlap)



        self.aff_graph[:,
        int(self.current_loc[0, 0] - self.overlap_increment[0]):int(self.current_loc[0, 0]- self.overlap_increment[0] + np.shape(blended_slice)[1]),
        int(self.current_loc[1, 0]):int(self.current_loc[1, 0]+np.shape(blended_slice)[2]),
        int(self.current_loc[2, 0]):int(self.current_loc[2, 0]+np.shape(blended_slice)[3])] = blended_slice

        return blended_slice

    def blend_axis1(self):
        # blends current slice with preivous slice on axis0

        ##Go back one overlap increment to find start      ##take start and add the shape to it
        prev_slice = self.aff_graph[:,
                     int(self.current_loc[0, 0]):int(self.current_loc[0, 0] + self.shape[0]),
                     int(self.current_loc[1, 0]- self.overlap_increment[1]):int(self.current_loc[1, 0]-self.overlap_increment[0] + self.shape[1]),
                     int(self.current_loc[2, 0]):int(self.current_loc[2, 0] + self.shape[2])]

        blended_slice = blend.blend(prev_slice, self.aff_block, "1+",self.overlap)

        self.aff_graph[:,
        int(self.current_loc[0, 0]):int(self.current_loc[0, 0]+np.shape(blended_slice)[1]),
        int(self.current_loc[1, 0] - self.overlap_increment[1]):int(self.current_loc[1, 0] - self.overlap_increment[1] + np.shape(blended_slice)[2]),
        int(self.current_loc[2, 0]):int(self.current_loc[2, 0]+np.shape(blended_slice)[3])] = blended_slice
        return blended_slice

    def blend_axis2(self):
        # blends current slice with preivous slice on axis0

        ##Go back one overlap increment to find start      ##take start and add the shape to it
        prev_slice = self.aff_graph[:,
                     int(self.current_loc[0, 0]):int(self.current_loc[0, 0] + self.shape[0]),
                     int(self.current_loc[1, 0]):int(self.current_loc[1, 0] + self.shape[1]),
                     int(self.current_loc[2, 0]- self.overlap_increment[2]):int(self.current_loc[2, 0]- self.overlap_increment[2] + self.shape[2])]

        blended_slice = blend.blend(prev_slice, self.aff_block, "2+",self.overlap)

        self.aff_graph[:,
        int(self.current_loc[0, 0]):int(self.current_loc[0, 0]+np.shape(blended_slice)[1]),
        int(self.current_loc[1, 0]):int(self.current_loc[1, 0]+np.shape(blended_slice)[2]),
        int(self.current_loc[2, 0]- self.overlap_increment[2]):int(self.current_loc[2, 0]- self.overlap_increment[2]+np.shape(blended_slice)[3])] = blended_slice
        return blended_slice

    def half_axis1_blend_axis0(self):
        #takes lower half on axis 2 of section and blends it with previous slice

        #first save current loc because it will be altered
        temp_loc=self.current_loc
        temp_aff_block=self.aff_block
        temp_shape=self.shape

        #redefine
        self.shape=np.asarray([self.shape[0],int(self.shape[1]/2),self.shape[2]])
        self.current_loc=np.asarray([self.current_loc[0],self.current_loc[1]+self.shape[1],self.current_loc[2]])
        self.aff_block=self.aff_block[:,:,-self.shape[1]:,:]

        #blend under new conditions
        blended_slice=self.blend_axis0()


        #take blended half and add to the block
        temp_aff_block[:, :, -self.shape[1]:,:] = blended_slice[:, -self.shape[0]:, :, :]

        #reassign all values
        self.current_loc=temp_loc
        self.aff_block=temp_aff_block
        self.shape=temp_shape
        return blended_slice

    def half_axis2_blend_axis0(self):
        # takes lower half on axis 3 of section and blends it with previous slice

        # first save current loc because it will be altered
        temp_loc = self.current_loc
        temp_aff_block = self.aff_block
        temp_shape = self.shape

        # redefine
        self.shape = np.asarray([self.shape[0], self.shape[1],int(self.shape[2] / 2)])
        self.current_loc = np.asarray([self.current_loc[0],self.current_loc[1],self.current_loc[2] + self.shape[2]])
        self.aff_block = self.aff_block[:, :,:, -self.shape[2]:]

        # blend under new conditions
        blended_slice = self.blend_axis0()

        # take blended half and add to the block
        temp_aff_block[:,:,:, -self.shape[2]:] = blended_slice[:,-self.shape[0]:, :, :]

        # reassign all values
        self.current_loc = temp_loc
        self.aff_block = temp_aff_block
        self.shape = temp_shape

        return blended_slice

    def half_axis2_blend_axis1(self):
        # takes lower half on axis 3 of section and blends it with previous slice

        # first save current loc because it will be altered
        temp_loc = self.current_loc
        temp_aff_block = self.aff_block
        temp_shape = self.shape

        # redefine
        self.shape = np.asarray([self.shape[0], self.shape[1], int(self.shape[2] / 2)])
        self.current_loc = np.asarray([self.current_loc[0], self.current_loc[1],self.current_loc[2] + self.shape[2]])
        self.aff_block = self.aff_block[:, :, :, -self.shape[2]:]

        # blend under new conditions
        blended_slice = self.blend_axis1()

        # take blended half and add to the block
        temp_aff_block[:, :, :, -self.shape[2]:] = blended_slice[:, :, -self.shape[1]:, :]

        # reassign all values
        self.current_loc = temp_loc
        self.aff_block = temp_aff_block
        self.shape = temp_shape

    def half_axis2_half_axis1_blend_axis0_blend_axis1(self):
        #blend half of lower half of block on axis 0

        # first save current loc because it will be altered
        temp_loc = self.current_loc
        temp_aff_block = self.aff_block
        temp_shape = self.shape

        # redefine
        self.shape = np.asarray([self.shape[0], self.shape[1], int(self.shape[2] / 2)])
        self.current_loc = np.asarray([self.current_loc[0], self.current_loc[1],self.current_loc[2] + self.shape[2]])
        self.aff_block = self.aff_block[:, :, :, -self.shape[2]:]

        #now we have lower half so call half axis1 blend
        self.half_axis1_blend_axis0()
        #then blend on axis 1
        blended_slice=self.blend_axis1()

        #at this point bottom half is completely blended and we must appened it to untouched top half

        #take blended half and add to the block
        temp_aff_block[:, :,:, -self.shape[2]:] = blended_slice[:, :, -self.shape[1]:, :]


        # reassign all values
        self.current_loc = temp_loc
        self.aff_block = temp_aff_block
        self.shape = temp_shape
        return blended_slice

    def blend_block(self):

        #row is itterated first then column then layer
        #axis1=row axis2=col axis3=layer



        #if first one add to corner
        if(self.counter==0):
            self.aff_graph[:,0:self.shape[0],0:self.shape[1],0:self.shape[2]]=self.aff_block

        #if in first column and layer
        if(self.dict["+0"]==False and self.dict["+1"]==True and self.dict["+2"]==True):
            self.blend_axis0()


        #if in first row but not first column and in first layer
        if(self.dict["+0"]==True and self.dict["+1"]==False and self.dict["+2"]==True):
            self.blend_axis1()

        #if not in first row or column but in first layer
        if(self.dict["+0"]==False and self.dict["+1"]==False and self.dict["+2"]==True):
            self.half_axis1_blend_axis0()
            self.blend_axis1()

        #if first row and column but not first layer
        if (self.dict["+0"] == True and self.dict["+1"] == True and self.dict["+2"] == False):
            self.blend_axis2()

        #if not first row but fist column and not first layer
        if (self.dict["+0"] == False and self.dict["+1"] == True and self.dict["+2"] == False):
            self.half_axis2_blend_axis0()
            self.blend_axis2()

        #if first row but not first column or layer
        if (self.dict["+0"] == True and self.dict["+1"] == False and self.dict["+2"] == False):
            self.half_axis2_blend_axis1()
            self.blend_axis2()

        #if not first row, column or layer
        if (self.dict["+0"] == False and self.dict["+1"] == False and self.dict["+2"] == False):
            #blend bottom half same way as "if not in first row or column but in first layer" then blend axis2
            self.half_axis2_half_axis1_blend_axis0_blend_axis1()
            self.blend_axis2()

    def crop_aff(self):
        print(np.shape(self.aff_graph))
        self.aff_graph = np.asarray(self.aff_graph)[:, 0:self.raw_shape[0] - 1, 0:self.raw_shape[1] - 1, 0:self.raw_shape[2] - 1]
        print(np.shape(self.aff_graph))












