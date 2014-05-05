# Copyright (C) 2014 ETH Zurich, Institute for Astronomy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/.


"""
Tests for `workflow` module.
"""
#from __future__ import print_function, division, absolute_import, unicode_literals
import os
import ConfigParser 
import shutil
import PynPoint


class TestWorkflow(object):

    def setup(self):
        #prepare unit test. Load data etc
        print("setting up " + __name__)
        self.data_dir = (os.path.dirname(__file__))+'/test_data/'
        configfile_temp = self.data_dir+'PynPoint_test_v001.config'
        configfile_temp2 = self.data_dir+'PynPoint_test_v002.config'

        config = ConfigParser.ConfigParser()
        config.optionxform = str

        config2 = ConfigParser.ConfigParser()
        config2.optionxform = str


        self.configfile = self.data_dir+'PynPoint_test_v001_out.config'
        self.configfile2 = self.data_dir+'PynPoint_test_v002_out.config'

        # workdir = /Users/amaraa/Work/Active_Projects/PynPoint_v1_5/test/test_data/workspace_temp
        # datadir = /Users/amaraa/Work/Active_Projects/PynPoint_v1_5/test/

        config.read(configfile_temp)
        config.set('workspace','workdir',self.data_dir+'workspace_temp')
        config.set('workspace','datadir',os.path.dirname(__file__)+'/')

        config2.read(configfile_temp2)
        config2.set('workspace','workdir',self.data_dir+'workspace_temp')
        config2.set('workspace','datadir',os.path.dirname(__file__)+'/')


        cgfile = open(self.configfile,'w')
        config.write(cgfile)
        cgfile.close()

        cgfile2 = open(self.configfile2,'w')
        config2.write(cgfile2)
        cgfile2.close()


        self.config = ConfigParser.ConfigParser()
        self.config.optionxform = str
        self.config.read(self.configfile)


        self.wf1 = wf_init = PynPoint.workflow()
        self.wf2 = wf_init = PynPoint.workflow()
        self.wf2._init_config(self.config)
        self.test_kwargs = {'para_sort': 'True', 'inner_pix': 'False', 'edge_size': '1.0', 
                            'ran_sub': 'False', 'resize': 'True', 'F_final': '2', 
                            'cent_size': '0.2', 'F_int': '4', 'cent_remove': 'True', 'recent': 'False'}
        
        

        pass
        

    def test__init(self):
        # wf_init = PynPoint.workflow()
        assert self.wf1.obj_type == 'PynPoint_workflow'
        assert self.wf1.module_string == 'module'

    def test_init_config_instance(self):
        self.wf1._init_config(self.config)
        # self.config
        assert hasattr(self.wf1,'config')
        assert self.wf1.config == self.config
        # assert 1==1
        
    def test_init_config_file(self):
        self.wf1._init_config(self.configfile)        
        # self.configfile
        assert hasattr(self.wf1,'config')
        assert self.wf1.config._sections == self.config._sections
        
    def test_setup_workspace(self):
        self.wf2._setup_workspace()
        assert os.path.exists(self.wf2.config.get('workspace','workdir'))
        assert os.path.exists(self.wf2.config.get('workspace','workdir')+'/wf.config')
        assert os.path.exists(self.wf2.config.get('workspace','workdir')+'/book_keeping.txt')
        
    def test_tidyup(self):
        self.wf2._setup_workspace()
        self.wf2._tidyup()
        assert os.path.exists(self.wf2.config.get('workspace','workdir')+'/book_keeping.txt')
        
    def test_get_keyword_options(self):
        kwargs = self.wf2._get_keyword_options('module1')
        print(dict(kwargs))
        # assert dict(kwargs) == self.test_kwargs
    
    def test_run_images_mod(self):
        self.wf2._setup_workspace()
        temp_images = self.wf2._run_images_mod('module1')
        assert hasattr(temp_images,'im_arr')
        assert temp_images.im_arr.shape == (4,292,292)
        
    def test_run_basis_mod(self):
        self.wf2._setup_workspace()
        temp_basis = self.wf2._run_basis_mod('module2')
        assert hasattr(temp_basis,'psf_basis')
        assert temp_basis.im_arr.shape == (4,292,292)

    def test_runmods(self):
        self.wf2._setup_workspace()
        self.wf2._runmods()


    def test_overall(self):
        ws = PynPoint.workflow.run(self.configfile)

    def test_overall2(self):
        ws = PynPoint.workflow.run(self.configfile2)


        
        



    # def test_something(self):
    #     x = 1
    #     assert x==1



    def teardown(self):
        dirname = self.data_dir+'workspace_temp'
        #tidy up
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
            
        if os.path.isfile(self.configfile):
            os.remove(self.configfile)
             
                
        print("tearing down " + __name__)
        pass