/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2018 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifdef __PLUMED_HAS_LIBTORCH

#include "bias/Bias.h"
#include "core/PlumedMain.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/Atoms.h"
#include "tools/Grid.h"
#include "tools/IFile.h"

#include <torch/torch.h>

#include "tools/Communicator.h"

using namespace std;


namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS NN_VES
/*
Work in progress
*/
//+ENDPLUMEDOC

//aux function to sum two vectors element-wise
template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b){
    assert(a.size() == b.size());
    std::vector<T> result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}
//aux function to multiply all elements of a vector with a scalar
template <class T, class Q>
std::vector <T> operator* (const Q c, std::vector <T> A){
    std::transform (A.begin (), A.end (), A.begin (),
                 std::bind1st (std::multiplies <T> () , c)) ;
    return A ;
}
std::vector<float> tensor_to_vector(const torch::Tensor& x) {
    return std::vector<float>(x.data<float>(), x.data<float>() + x.numel());
}
std::vector<double> tensor_to_vector_d(const torch::Tensor& x) {
    return std::vector<double>(x.data<float>(), x.data<float>() + x.numel());
} 
float tensor_to_scalar(const torch::Tensor& x){
    return x.item<float>();
}
// exp_added(expsum,expvalue)
// expsum=log(exp(expsum)+exp(expvalue)
inline void exp_added(double& expsum,double expvalue)
{
    if(expsum>expvalue)
	expsum=expsum+std::log(1.0+exp(expvalue-expsum));
    else
	expsum=expvalue+std::log(1.0+exp(expsum-expvalue));
}

//enumeration for activation functions
enum Activation { SIGMOID, TANH, RELU,  ELU, LINEAR};
 
Activation set_activation(const std::string& a)
{
	if(a=="SIGMOID"||a=="sigmoid"||a=="Sigmoid")
		return 	Activation::SIGMOID;
	if(a=="TANH"||a=="tanh"||a=="Tanh")
		return 	Activation::TANH;
	if(a=="ELU"||a=="elu"||a=="Elu"||a=="eLU")
		return 	Activation::ELU;
	if(a=="RELU"||a=="relu"||a=="Relu"||a=="ReLU")
		return 	Activation::RELU;
	if(a=="LINEAR"||a=="linear"||a=="Linear")
		return 	Activation::LINEAR;
	std::cerr<<"ERROR! Can't recognize the activation function "+a<<std::endl;
	exit(-1);
}

inline torch::Tensor activate(torch::Tensor x, torch::nn::Linear l, Activation f)
{
    switch (f) {
    case LINEAR:
      return l->forward(x);	
      break;
    case ELU:
      return torch::elu(l->forward(x));
      break;
    case RELU:
      return torch::relu(l->forward(x)); 
      break;
    case SIGMOID:
      return torch::sigmoid(l->forward(x));
      break;
    case TANH:
      return torch::tanh(l->forward(x));
      break;
    default:
      throw std::invalid_argument("Unknown activation function");
      break;
    }
}

// NEURAL NETWORK MODULE
struct Net : torch::nn::Module {
  //constructor //
  Net( vector<int> nodes, vector<bool> periodic, std::string activ ) : _layers() {
    //get number of hidden layers 
    _hidden=nodes.size() - 2;
    //check wheter to enforce periodicity
    _periodic=periodic;
    for(int i=0;i<nodes[0];i++)
      if(_periodic[i])	
        nodes[0]++;
    //save activation function for hidden layers
    _activ=set_activation(activ);
    //normalize later, using the method
    _normalize=false;
    //register modules
    for(int i=0; i<_hidden; i++)
        _layers.push_back( register_module("fc"+to_string(i+1), torch::nn::Linear(nodes[i], nodes[i+1])) );
    //register output layer
    _out = register_module("out", torch::nn::Linear(nodes[_hidden], nodes[_hidden+1]));
  }

  ~Net() {}

  //set range to normalize input 
  void setRange(vector<string> m, vector<string> M){
    _normalize=true;
    vector<float> mean, inv_range;
    for(unsigned i=0;i<m.size();i++){
      double max,min;
      Tools::convert(m[i],min);
      Tools::convert(M[i],max);
      mean.push_back( (max+min)/2. );
      inv_range.push_back( 1./( (max-min)/2.) ) ;
    } 
    _mean = torch::tensor(mean).view({1,m.size()});
    _inv_range = torch::tensor(inv_range).view({1,m.size()});
  }

  //forward operation
  torch::Tensor forward(torch::Tensor x) {
    //enforce periodicity (encode every input x into {cos(x), sin(x)} ): works only if all of them are periodic TODO
    if(_periodic[0]) //size(0): number of elements in batch - size(1): size of the input 
        x = at::stack({at::sin(x),at::cos(x)},1).view({x.size(0),2*x.size(1)});
    //normalize input (given range) 
    else if(_normalize)
	x=(x-_mean)*_inv_range;
    //now propagate
    for(unsigned i=0; i<_layers.size(); i++)
	x = activate(x,_layers[i],_activ);
        //x = torch::elu(_layers[i]->forward(x));
    x = _out->forward(x);
    return x;
  }

  /*--class members--*/
  int 			_hidden;
  bool			_normalize;
  vector<bool>		_periodic;
  vector<float>		_min, _max;
  torch::Tensor		_mean, _inv_range;
  vector<torch::nn::Linear> _layers;
  torch::nn::Linear 	_out = nullptr;
  Activation 		_activ;
};

class NeuralNetworkVes : public Bias {
private:

/*--MPI Setup--*/
  unsigned 		mpi_num;
  unsigned 		mpi_rank;
/*--neural_network_setup--*/
  unsigned 		nn_dim;
  vector<int> 		nn_nodes;
  shared_ptr<Net>	nn_model;
  shared_ptr<torch::optim::Optimizer> nn_opt; 
/*--parameters and options--*/
  float 		o_beta;
  int	 		o_stride;
  int	 		o_print;
  int			o_target;
  int 			o_tau;
  float 		o_lrate;
  float			o_gamma;
  vector<bool>		o_periodic;
  float			o_decay;
  float			o_adaptive_decay; 
  bool			o_coft;
/*--counters--*/
  int			c_iter;
  int			c_start_from;
  bool			c_is_first_step;
  double		c_lr_scaling;
  bool			c_static_model;
/*--grids--*/
  std::unique_ptr<Grid> grid_bias,grid_target_ds,grid_bias_hist,grid_fes; 
  vector<string>	g_min, g_max; 
  vector<float>		g_ds;
  vector<unsigned>	g_nbins;
  vector<unsigned>	g_counter;
  double		g_target_norm;
/*--reweight--*/
  float 		r_ct; 
  float 		r_bias;
/*--gradients vectors--*/
  vector<vector<float>>	g_;
  vector<vector<float>>	g_mean;
  vector<vector<float>>	g_target;
  vector<torch::Tensor> g_tensor;
/*--outputvalues--*/
  Value*		v_kl;
  Value*		v_rct;
  Value*		v_rbias;
  Value*		v_ForceTot2;
  Value*		v_lr;
/*--methods-*/
  void 			update_coeffs();

public:
  explicit NeuralNetworkVes(const ActionOptions&);
  ~NeuralNetworkVes() {};
  void calculate();
  std::unique_ptr<GridBase> createGridFromFile(const std::string&,const std::vector<Value*>&, const std::string&, const std::vector<std::string>&,const std::vector<std::string>&,const std::vector<unsigned>&,bool,bool);
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(NeuralNetworkVes,"NN_VES")

void NeuralNetworkVes::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");

  keys.add("compulsory","NODES","neural network architecture");
  keys.add("optional","LRATE","the step used for the minimization of the functional");
  keys.add("optional","OPTIM","choose the optimizer");
  keys.add("optional","ACTIVATION","activation function for hidden layers");
  keys.add("optional","BETA1","b1 coeff of ADAM");
  keys.add("optional","BETA2","b2 coeff of ADAM");

  keys.add("compulsory","GRID_MIN","min of the target distribution range");
  keys.add("compulsory","GRID_MAX","max of the target distribution range");
  keys.add("compulsory","GRID_BIN","number of bins");

  keys.add("optional","TAU_KL","exponentially decaying average for KL");
  keys.add("optional","DECAY","decay constant for learning rate");
  keys.add("optional","ADAPTIVE_DECAY","whether to adapt lr to KL under a threshold");

  keys.add("optional","TEMP","temperature of the simulation");
  keys.add("optional","GAMMA","gamma value for well-tempered distribution");

  keys.add("optional","AVE_STRIDE","the stride for the update of the bias");
  keys.add("optional","PRINT_STRIDE","the stride for printing the bias (iterations)");
  keys.add("optional","TARGET_STRIDE","the stride for updating the iterations (iterations)");

  keys.add("optional","RESTART_FROM","iteration to restart from");
  keys.add("optional","COLVAR_FILE","colvar filename for restart");
  keys.add("optional","NN_STATIC_BIAS","load model parameters");

  keys.addFlag("SERIAL",false,"run without grid parallelization");
  keys.addFlag("CALC_RCT",false,"compute c(t)");

  componentsAreNotOptional(keys);
  useCustomisableComponents(keys); //needed to have an unknown number of components/
  keys.addOutputComponent("_bias","default","one or multiple instances of this quantity can be referenced elsewhere in the input file.");
  keys.addOutputComponent("kl","default","kl divergence between bias and target");
  keys.addOutputComponent("rct","default","c(t) term");
  keys.addOutputComponent("rbias","default","bias-c(t)");
  keys.addOutputComponent("force2","default","total force");
  keys.addOutputComponent("lrscale","default","scaling factor for learning rate");
}

NeuralNetworkVes::NeuralNetworkVes(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao),
  c_is_first_step(true),
  g_target_norm(0.)
{
  //for debugging TODO remove?
  //torch::manual_seed(0);

  /*--NN OPTIONS--*/
  //get # of inputs (CVs)
  nn_dim=getNumberOfArguments();
  //parse the NN architecture
  parseVector("NODES",nn_nodes);
  //add input and output nodes
  nn_nodes.insert(nn_nodes.begin(), nn_dim);
  nn_nodes.push_back(1);

  /*--TEMPERATURE--*/  
  double temp=0;
  parse("TEMP",temp);
  double Kb = plumed.getAtoms().getKBoltzmann();
  double KbT = plumed.getAtoms().getKBoltzmann()*temp;
  if(KbT<=0){
    KbT=plumed.getAtoms().getKbT();
    plumed_massert(KbT>0,"your MD engine does not pass the temperature to plumed, you must specify it using TEMP");
  }
  o_beta = 1./KbT;

  /*--PARAMETERS--*/
  // update stride
  o_stride=500;
  parse("AVE_STRIDE",o_stride);
  // print stride
  o_print=1000;
  parse("PRINT_STRIDE",o_print);
  // update stride
  o_target=100;
  parse("TARGET_STRIDE",o_target);
 
  // check whether to use an exponentially decaying average for the calculation of KL
  o_tau=0;
  parse("TAU_KL",o_tau);
  if (o_tau>0)
    o_tau*=o_stride;
  // parse learning rate
  o_lrate=0.001;
  parse("LRATE",o_lrate); 
  c_lr_scaling=1.;
  // parse gamma
  o_gamma=0;
  parse("GAMMA",o_gamma);
  // check if args are periodic 
  o_periodic.resize(nn_dim);
  for (unsigned i=0; i<nn_dim; i++)
    o_periodic[i]=getPntrToArgument(i)->isPeriodic();

  /*--GRID OPTIONS--*/
  // range 
  g_min.resize(nn_dim);
  g_max.resize(nn_dim);
  parseVector("GRID_MIN",g_min);
  parseVector("GRID_MAX",g_max);
  // nbins
  g_nbins.resize(nn_dim);
  parseVector("GRID_BIN", g_nbins); 
  // gridoptions
  bool spline=true;
  bool sparsegrid=false;

  //MPI OPTIONS
  mpi_num=comm.Get_size();
  mpi_rank=comm.Get_rank();
  bool serial=false;
  parseFlag("SERIAL",serial);
  if (serial)
  {
    mpi_num=1;
    mpi_rank=0;
  }

  //coft
  o_coft=false;
  parseFlag("CALC_RCT",o_coft);

  /*--GRID SETUP--*/
  if(!getRestart()){
    // reset counters
    c_iter=0;
    c_start_from=0;
    //init grids 
    grid_bias.reset(new Grid(getLabel()+".bias",getArguments(),g_min,g_max,g_nbins,/*spline*/spline,true));
    grid_bias_hist.reset(new Grid(getLabel()+".hist",getArguments(),g_min,g_max,g_nbins,/*spline*/false,true));
    grid_fes.reset(new Grid(getLabel()+".fes",getArguments(),g_min,g_max,g_nbins,/*spline*/false,true));
    grid_target_ds.reset(new Grid(getLabel()+".target",getArguments(),g_min,g_max,g_nbins,/*spline*/false,true));
 
    //fill grids with initial values 
    for (Grid::index_t t=mpi_rank; t<grid_fes->getSize(); t+=mpi_num){
      grid_bias_hist->setValue(t,1e-6);
      grid_fes->setValue(t,0.);
    } 
  }else{ //if restart
    // WARNING: RESTART NOT IMPLEMENTED
    error("RESTART NOT IMPLEMENTED.");

    if( keywords.exists("RESTART_FROM")){
      parse("RESTART_FROM",c_start_from);
      c_iter=c_start_from;
    } else{
      //get iteration number from md engine
      c_iter=(getStep()/o_stride);
      //retrieve latest grid printed
      c_start_from=(c_iter/o_print)*o_print;
    }
    //create grids from files // LB this needs to be fixed with the new Grid/GridBase version
    //grid_bias=createGridFromFile(getLabel()+".bias",getArguments(),"bias.iter-"+to_string(c_start_from),g_min,g_max,g_nbins,sparsegrid,spline);
    //grid_bias_hist=createGridFromFile(getLabel()+".hist",getArguments(),"hist.iter-"+to_string(c_start_from),g_min,g_max,g_nbins,sparsegrid,false);
    //grid_fes=createGridFromFile(getLabel()+".fes",getArguments(),"fes.iter-"+to_string(c_start_from),g_min,g_max,g_nbins,sparsegrid,false);

    //if(o_target>0)
    //  grid_target_ds=createGridFromFile(getLabel()+".target",getArguments(),"target.iter-"+to_string(c_start_from),g_min,g_max,g_nbins,sparsegrid,false);
    //else
    //  grid_target_ds.reset(new Grid(getLabel()+".target",getArguments(),g_min,g_max,g_nbins,/*spline*/false,true));

    //sync all . Not sure is mandatory but is no harm
     if(mpi_num>1)
       comm.Barrier();
  
  }

  unsigned nbins=1;
  //normalize target distribution (if not restarted or if uniform td)
  if(!getRestart() || o_target==0){
    vector<unsigned> bins=grid_target_ds->getNbin();
    for(auto&& b : bins)
      nbins *= b;
    for (Grid::index_t t=mpi_rank; t<grid_fes->getSize(); t+=mpi_num)
      grid_target_ds->setValue(t,1./nbins);
  } 

  //define counter for kl historam
  g_counter.resize( nbins );
  std::fill(g_counter.begin(), g_counter.end(), 0);
 
  /*--NEURAL NETWORK SETUP --*/
  //set the activation function
  std::string activation = "ELU";
  parse("ACTIVATION",activation);
  nn_model = std::make_shared<Net>(nn_nodes, o_periodic, activation);
  //set range for normalization
  nn_model->setRange(g_min, g_max);
  //select the optimizer
  std::string opt="ADAM";
  parse("OPTIM",opt);
  if (opt=="SGD")
    nn_opt = make_shared<torch::optim::SGD>(nn_model->parameters(), o_lrate);
  else if (opt=="NESTEROV"){
    torch::optim::SGDOptions opt(o_lrate);
    opt.nesterov(true);
    nn_opt = make_shared<torch::optim::SGD>(nn_model->parameters(), opt);
  }else if (opt=="RMSPROP")
    nn_opt = make_shared<torch::optim::RMSprop>(nn_model->parameters(), o_lrate);
  else if (opt=="ADAM"){
    //nn_opt = make_shared<torch::optim::Adam>(nn_model->parameters(), o_lrate);
    torch::optim::AdamOptions opt(o_lrate);
    float b1=0.9, b2=0.999;
    parse("BETA1",b1);
    parse("BETA2",b2);
    auto betas = std::make_tuple(b1,b2);
    opt.betas(betas);
    nn_opt = make_shared<torch::optim::Adam>(nn_model->parameters(), opt);
  }else if (opt=="ADAGRAD")
    nn_opt = make_shared<torch::optim::Adagrad>(nn_model->parameters(), o_lrate);
  else if (opt=="AMSGRAD"){
    torch::optim::AdamOptions opt(o_lrate);
    opt.amsgrad(true);
    nn_opt = make_shared<torch::optim::Adam>(nn_model->parameters(), opt);
  } else {
    cerr<<"ERROR! Can't recognize the optimizer: "+opt<<endl;
        exit(-1);
  }
  //parse the decay time for computing the KL
  o_decay=0;
  parse("DECAY",o_decay);
  if(o_decay>0.) 		//convert from decay time to multiplicative factor: lr = lr * o_decay
    o_decay=1-1/o_decay;

  //whether to adapt learning rate to kl divergence
  o_adaptive_decay=0.;
  parse("ADAPTIVE_DECAY",o_adaptive_decay);
   
  /*--CREATE AUXILIARY VECTORS--*/
  //dummy backward pass in order to have the grads defined
  vector<torch::Tensor> params = nn_model->parameters();
  torch::Tensor y = nn_model->forward( torch::rand({nn_dim}).view({1,nn_dim}) );
  y.backward();
  //Define auxiliary vectors to store gradients
  nn_opt->zero_grad();
  for (auto&& p : params ){
    vector<float> gg = tensor_to_vector( p.grad() );
    g_.push_back(gg);
    g_mean.push_back(gg);
    g_target.push_back(gg);
    g_tensor.push_back(p);
  }
  //reset mean grads
  for (unsigned i=0; i<g_.size()-1; i++){ 
    std::fill(g_[i].begin(), g_[i].end(), 0.);
    std::fill(g_mean[i].begin(), g_mean[i].end(), 0.);
    std::fill(g_target[i].begin(), g_target[i].end(), 0.);
  }
  /*--SET OUTPUT COMPONENTS--*/
  addComponent("force2"); componentIsNotPeriodic("force2");
  v_ForceTot2=getPntrToComponent("force2");
  addComponent("kl"); componentIsNotPeriodic("kl");
  v_kl=getPntrToComponent("kl");
   addComponent("lrscale"); componentIsNotPeriodic("lrscale");
  v_lr=getPntrToComponent("lrscale");
  if(o_coft){
    addComponent("rct"); componentIsNotPeriodic("rct");
    v_rct=getPntrToComponent("rct");
    addComponent("rbias"); componentIsNotPeriodic("rbias");
    v_rbias=getPntrToComponent("rbias");
  }

  /*--LOAD MODEL AND OPT PARAMETERS FROM RESTART*/
  if (getRestart()){
    shared_ptr<torch::serialize::InputArchive> check_model = make_shared<torch::serialize::InputArchive>();
    shared_ptr<torch::serialize::InputArchive> check_opt = make_shared<torch::serialize::InputArchive>();
    check_model->load_from("model_checkpoint.pt");
    check_opt->load_from("opt_checkpoint.pt");
    nn_model->load(*check_model);
    nn_opt->load(*check_opt);
  
    //retrieve learning rate from COLVAR file
    IFile ifile;
    ifile.link(*this);
    ifile.allowIgnoredFields(); 
    std::string colvar_file="COLVAR";
    parse("COLVAR_FILE",colvar_file);

    if (ifile.FileExist(colvar_file)){
      ifile.open(colvar_file);
      double time;
      while(ifile.scanField("time",time)){
        ifile.scanField(getLabel()+".lrscale",c_lr_scaling);
        ifile.scanField();
      }
      ifile.close();
      //set lr to model
      //dynamic_pointer_cast<torch::optim::Adam, torch::optim::Optimizer>(nn_opt)->options.learning_rate( o_lrate * c_lr_scaling );
      static_cast<torch::optim::AdamOptions&>(nn_opt->param_groups()[0].options()).lr( o_lrate * c_lr_scaling ); // LB new version 1.8
    } else
      error("The COLVAR file you want to read: " + colvar_file + ", cannot be found!"); 
    
  }

  getPntrToComponent("lrscale")->set( c_lr_scaling );

  /*--NN STATIC BIAS MODE --*/
  c_static_model=false;
  std::string model_file;
  parse("NN_STATIC_BIAS",model_file);
  if(!model_file.empty() ){
    c_static_model=true;
    shared_ptr<torch::serialize::InputArchive> check_model = make_shared<torch::serialize::InputArchive>();
    check_model->load_from(model_file);
    nn_model->load(*check_model);
  }
  /*--PARSING DONE --*/
  checkRead();

  /*--LOG INFO--*/
  log.printf("  ninputs: %d\n",nn_dim);
  log.printf("  Temperature T: %g\n",1./(Kb*o_beta));
  log.printf("  Beta (1/Kb*T): %g\n",o_beta);
  log.printf("  -- NEURAL NETWORK SETUP --\n");
  log.printf("  Architecture: "); for(auto&& n : nn_nodes) log.printf(" %d",n); log.printf("\n");
  log.printf("  Optimizer: %s\n",opt.c_str());
  log.printf("  Activation function: %s\n",activation.c_str());
  log.printf("  Learning Rate: %g\n",o_lrate);
  if(o_decay>0) log.printf("  - with a decaying constant, with multiplier: %g\n",o_decay);
  if(o_adaptive_decay>0) log.printf("  - only when the KL divergence is below: %g\n",o_adaptive_decay); 
  if(o_tau==0) log.printf("  KL divergence between biased and target is calculated using all data from the simulation.\n");
  else log.printf("  KL divergence between biased and target is calculated with an exponential decaying average with decay time of  %d (%d iterations)\n",o_tau*o_stride,o_tau);
  log.printf("  -- VES SETUP SETUP --\n");
  log.printf("  Stride : %d\n",o_stride);
  log.printf("  Target distribution : ");
  if(o_target==0) log.printf("UNIFORM");
  else log.printf("WELL-TEMPERED, with a recursive update every %d iterations\n",o_target);
  if (getRestart()) log.printf("  RESTART from iteration: %d\n",c_start_from);
  log.printf("  -- GRID SETTINGS --\n");
  log.printf("  TODO \n");
  log.printf("  -- MPI SETTINGS --\n");
  log.printf("  Num processes: %d\n",mpi_num);
  if (c_static_model) log.printf("  -- STATIC BIAS --\n"); 
}

void NeuralNetworkVes::calculate() {
  double bias_pot=0;
  double tot_force2=0;
  vector<double> der(nn_dim);

  //check whether to update the bias or use the one stored in the grid 
  bool static_bias = false; 
  if(c_lr_scaling < 5.e-5) static_bias=true;
  //check which operations to do
  bool do_stride=false, do_coft=false, do_update_target=false, do_print=false, do_save_bias=false;
  if(!c_static_model){
    do_stride = ( getStep() % o_stride == 0 );
    do_coft = o_coft;
    if (do_stride && !static_bias ){
      do_print = ( c_iter % o_print == 0 );
      do_update_target = ( o_target > 0 && c_iter % o_target == 0 );
      do_save_bias = ( do_print || do_update_target );
    }
  }

if(!static_bias){
  //get current CVs
  vector<float> current_S(nn_dim);
  for(unsigned i=0; i<nn_dim; i++)
    current_S[i]=getArgument(i);
  //get current CVs (double, to check grid error)
  vector<double> cv(nn_dim);
  for(unsigned i=0; i<nn_dim; i++)
      cv[i]=getArgument(i); 
  //convert current CVs into torch::Tensor
  torch::Tensor input_S = torch::tensor(current_S).view({1,nn_dim});
  input_S.set_requires_grad(true);
  //propagate to get the bias
  nn_opt->zero_grad();
  auto output = nn_model->forward( input_S );
  bias_pot = output.item<float>();
  //backprop to get forces
  output.backward();
  der = tensor_to_vector_d( input_S.grad() );
  //accumulate gradients
  vector<torch::Tensor> p = nn_model->parameters();
  for (unsigned i=0; i<p.size(); i++){
    vector<float> gg = tensor_to_vector( p[i].grad() );
    g_mean[i] = g_mean[i] + gg;
  }
  //accumulate histogram for biased distribution
  //vector<double> s_double(current_S.begin(), current_S.end());
  Grid::index_t current_index = grid_bias_hist->getIndex(cv);
  g_counter[current_index] ++; 
/*
  //get current weight
  float weight=getStep()+1;
  if (o_tau>0 ) //&& weight>o_tau)
    weight=o_tau;

  for (Grid::index_t i=0; i<grid_bias_hist->getSize(); i+=1){
    double h = grid_bias_hist->getValue(i);
    if(i==current_index)
      grid_bias_hist->addValue(i,(1.-h)/weight);
    else
      grid_bias_hist->addValue(i,(-h)/weight);
  }
*/
  /*--UPDATE PARAMETERS--*/
  if( do_stride ){
    /**Biased ensemble contribution**/
    //normalize average gradient
    for (unsigned i=0; i<g_mean.size(); i++)
      g_mean[i] = -(1./o_stride) * g_mean[i];

    /**Target distribution contribution**/
    for (Grid::index_t i=mpi_rank; i<grid_fes->getSize(); i+=mpi_num){
      //scan over grid //TODO check conversion from double to float
      vector<double> point_S=grid_target_ds->getPoint(i);
      vector<float> target_S(point_S.begin(),point_S.end());
      torch::Tensor input_S_target = torch::tensor(target_S).view({1,nn_dim});
      input_S_target.set_requires_grad(true);
      nn_opt->zero_grad();
      output = nn_model->forward( input_S_target );
      output.backward();
//      vector<double> force=tensor_to_vector_d( input_S.grad() );
//      grid_bias->setValueAndDerivatives(i,output.item<double>(),force);
      p = nn_model->parameters();
      for(unsigned j=0; j<p.size()-1; j++){
        vector<float> gg = tensor_to_vector( p[j].grad() );
        gg = grid_target_ds->getValue(i) * gg;
        g_target[j] = g_target[j] + gg;
      }
    }
    if(mpi_num>1){
      for(unsigned j=0; j<p.size()-1; j++)
        comm.Sum(g_target[j]);
    }

    //reset gradients
    nn_opt->zero_grad();

    /**Assign new gradient and update coefficients**/
    for (unsigned i=0; i<g_.size()-1; i++){  //until size-1 since we do not want to update the bias of the output layer
	//bias-target
	g_[i]=g_mean[i]+g_target[i];
        //vector to Tensor
        g_tensor[i] = torch::tensor(g_[i]).view( nn_model->parameters()[i].sizes() );
        //assign tensor to derivatives
        //nn_model->parameters()[i].grad() = g_tensor[i].detach();
        nn_model->parameters()[i].mutable_grad() = g_tensor[i].detach(); // LB new version 1.8
        //reset mean grads
        std::fill(g_[i].begin(), g_[i].end(), 0.);
        std::fill(g_mean[i].begin(), g_mean[i].end(), 0.);
        std::fill(g_target[i].begin(), g_target[i].end(), 0.);
    }
    //update the parameters
    if( !c_is_first_step )
      nn_opt->step();

    /*--COMPUTE REWEIGHT FACTOR--*/ 
    g_target_norm=0;
  if( do_coft ){
    double log_sumebv=-1.0e38;
    //loop over grid
    for (Grid::index_t i=mpi_rank; i<grid_fes->getSize(); i+=mpi_num){
      double log_target = std::log( grid_target_ds->getValue(i) );	        
      double log_ebv = o_beta * grid_bias->getValue(i) + log_target;     	//beta*V(s)+log p(s)
      if(i==mpi_rank) log_sumebv = log_ebv;				//sum exp with previous ones (see func. exp_added)
      else exp_added(log_sumebv,log_ebv);
      g_target_norm += grid_target_ds->getValue(i);
    }
    if(mpi_num>1){
      comm.Sum(g_target_norm);
      std::vector<double> all_log_sumebv(mpi_num,0);
      if(mpi_rank==0){
        comm.Allgather(log_sumebv,all_log_sumebv);
        comm.Bcast(all_log_sumebv,0);
        log_sumebv=all_log_sumebv[0];
        for(unsigned i=1;i<mpi_num;++i)
          exp_added(log_sumebv,all_log_sumebv[i]);
      }
    }
    //compute c(t)
    r_ct = (log_sumebv-std::log(g_target_norm))/o_beta;
    getPntrToComponent("rct")->set(r_ct); 
  }

    /*--SAVE BIAS INTO THE GRID--*/
  if( do_save_bias ){ 
    for (Grid::index_t i=0; i<grid_fes->getSize(); i+=1){
      vector<double> point_S=grid_target_ds->getPoint(i);
      vector<float> target_S(point_S.begin(),point_S.end());
      torch::Tensor input_S_target = torch::tensor(target_S).view({1,nn_dim});
      input_S_target.set_requires_grad(true);
      nn_opt->zero_grad();
      output = nn_model->forward( input_S_target );
      output.backward();
      vector<double> force=tensor_to_vector_d( input_S_target.grad() );
      grid_bias->setValueAndDerivatives(i,output.item<double>(),force);
    }
  }

    //--COMPUTE KL--
    //get current weight
    float weight=getStep()+1;
    if (o_tau>0 ) //&& weight>o_tau)
      weight=o_tau;

    double bias_norm=0;
    for (Grid::index_t i=0; i<grid_bias_hist->getSize(); i+=1){
      double h = grid_bias_hist->getValue(i);
      double new_value = h+(g_counter[i]-o_stride*h)/weight;
      bias_norm+=new_value;
      grid_bias_hist->setValue(i,new_value);
      if(!o_coft) g_target_norm += grid_target_ds->getValue(i);
    }
    std::fill(g_counter.begin(), g_counter.end(), 0); 
/*
    //normalize bias histogram
    double bias_norm=0;
    for (Grid::index_t i=mpi_rank; i<grid_fes->getSize(); i+=mpi_num){
      bias_norm += grid_bias_hist->getValue(i);
      if(!o_coft) target_norm += grid_target_ds->getValue(i);
    }
    if(mpi_num>1){
      comm.Sum(bias_norm);
      if(!o_coft) comm.Sum(target_norm);
    }
*/
    //compute kl
    double kl=0;
    for (Grid::index_t i=mpi_rank; i<grid_bias_hist->getSize(); i+=mpi_num)
      kl+= (grid_bias_hist->getValue(i) / bias_norm) * std::log( ( grid_bias_hist->getValue(i) / bias_norm ) / ( grid_target_ds->getValue(i) / g_target_norm ) );
      //kl+= ( grid_target_ds->getValue(i) / target_norm ) * std::log( ( grid_target_ds->getValue(i) / target_norm ) / ( grid_bias_hist->getValue(i) / bias_norm )  );
    
    if(mpi_num>1)
      comm.Sum(kl);
    getPntrToComponent("kl")->set(kl);

  if( !c_is_first_step ){
    /*--LEARNING RATE DECAY--*/
    if(o_decay>0){
      //if adaptive: rescale it only when the KL is below a threshold
      if(o_adaptive_decay==0 || kl<o_adaptive_decay){
        c_lr_scaling*= o_decay;
	      double current_lr=o_lrate*c_lr_scaling; 
	      //dynamic_pointer_cast<torch::optim::Adam, torch::optim::Optimizer>(nn_opt)->options.learning_rate( current_lr );
        static_cast<torch::optim::AdamOptions&>(nn_opt->param_groups()[0].options()).lr( current_lr ); // LB new version 1.8
  
        getPntrToComponent("lrscale")->set(c_lr_scaling);
      }
    }

    /*--UPDATE TARGET DISTRIBUTION--*/ 
    float sum_exp_beta_F = 0;
    if( do_update_target ){
      //compute new estimate of the fes
      for (Grid::index_t i=0; i<grid_fes->getSize(); i+=1){
        grid_fes->setValue( i, - grid_bias->getValue(i) + (1./o_gamma) * grid_fes->getValue(i) );
        float exp_beta_F = std::exp( (-o_beta/o_gamma) * grid_fes->getValue(i) ); 
	      sum_exp_beta_F += exp_beta_F;
        grid_target_ds->setValue(i, exp_beta_F);
      }
      grid_target_ds->scaleAllValuesAndDerivatives( 1./sum_exp_beta_F );
    }

    if( do_print ){
    /*--PRINT GRIDS TO FILE--*/
      OFile ofile;
      ofile.link(*this);
      ofile.enforceBackup();

      if(o_target > 0){
        ofile.open( "target.iter-"+to_string(c_iter) );
        grid_target_ds->writeToFile(ofile);
        ofile.close();
        ofile.clearFields();
      }

      ofile.open( "fes.iter-"+to_string(c_iter) );
      grid_fes->writeToFile(ofile);
      ofile.close();
      ofile.clearFields();

      ofile.open( "bias.iter-"+to_string(c_iter) );
      grid_bias->writeToFile(ofile);
      ofile.close();
      ofile.clearFields();

      ofile.open( "hist.iter-"+to_string(c_iter) );
      grid_bias_hist->writeToFile(ofile);
      ofile.close();

    /*--SAVE CHECKPOINT --*/
      shared_ptr<torch::serialize::OutputArchive> check_model = make_shared<torch::serialize::OutputArchive>();
      shared_ptr<torch::serialize::OutputArchive> check_opt = make_shared<torch::serialize::OutputArchive>();
      nn_model->save(*check_model);
      nn_opt->save(*check_opt);
      check_model->save_to("model_checkpoint.pt");
      check_opt->save_to("opt_checkpoint.pt");
    }
  }

   /*--INCREMENT COUNTER--*/
   c_iter++; 

 }
//if static bias:
} else {
  //get current CVs
  vector<double> cv(nn_dim);
  for(unsigned i=0; i<nn_dim; i++)
    cv[i]=getArgument(i);
  bias_pot = grid_bias->getValueAndDerivatives(cv,der); 
  //accumulate histogram for kl 
  Grid::index_t current_index = grid_bias_hist->getIndex(cv);
  g_counter[current_index] ++;
  //compute KL every stride
  if(do_stride){
  //--COMPUTE KL--
  //get current weight
  float weight=getStep()+1;
  if (o_tau>0 ) //&& weight>o_tau)
    weight=o_tau;

  double bias_norm=0;
  //g_target_norm=0;
  for (Grid::index_t i=0; i<grid_bias_hist->getSize(); i+=1){
    double h = grid_bias_hist->getValue(i);
    double new_value = h+(g_counter[i]-o_stride*h)/weight;
    bias_norm+=new_value;
    grid_bias_hist->setValue(i,new_value);
    //if(!o_coft) g_target_norm += grid_target_ds->getValue(i);
  }
  std::fill(g_counter.begin(), g_counter.end(), 0); 
  //compute kl
  double kl=0;
  for (Grid::index_t i=mpi_rank; i<grid_bias_hist->getSize(); i+=mpi_num)
    kl+= (grid_bias_hist->getValue(i) / bias_norm) * std::log( ( grid_bias_hist->getValue(i) / bias_norm ) / ( grid_target_ds->getValue(i) / g_target_norm ) );

  if(mpi_num>1)
    comm.Sum(kl);
  getPntrToComponent("kl")->set(kl);  
  } 
}
  //set bias
  setBias(bias_pot);
  if(do_coft){    
    r_bias = bias_pot-r_ct;
    getPntrToComponent("rbias")->set(r_bias);
  }
  //set forces
  for (unsigned i=0; i<nn_dim; i++){
    tot_force2+=pow(der[i],2);
    setOutputForce(i,-der[i]); //be careful of minus sign
  }
  v_ForceTot2->set(tot_force2); 

  if(c_is_first_step){
    c_is_first_step=false;
    //save model if STATIC
    if(c_static_model){
     for (Grid::index_t i=0; i<grid_fes->getSize(); i+=1){
      vector<double> point_S=grid_target_ds->getPoint(i);
      vector<float> target_S(point_S.begin(),point_S.end());
      torch::Tensor input_S_target = torch::tensor(target_S).view({1,nn_dim});
      input_S_target.set_requires_grad(true);
      nn_opt->zero_grad();
      auto output = nn_model->forward( input_S_target );
      output.backward();
      vector<double> force=tensor_to_vector_d( input_S_target.grad() );
      grid_bias->setValueAndDerivatives(i,output.item<double>(),force);
     }
      OFile ofile;
      ofile.link(*this);
      ofile.enforceBackup();

      ofile.open( "bias.iter-"+to_string(c_iter) );
      grid_bias->writeToFile(ofile);
      ofile.close();
      ofile.clearFields();

    } //end static
  } //end first step
} //end compute


std::unique_ptr<GridBase> NeuralNetworkVes::createGridFromFile(const std::string& funcl,const std::vector<Value*>& args, const std::string& filename, const std::vector<std::string>& g_min,const std::vector<std::string>& g_max,const std::vector<unsigned>& g_nbins,bool sparsegrid,bool spline)
{
    PLMD::IFile gridfile;
    gridfile.link(*this);
    if(gridfile.FileExist(filename))
      gridfile.open(filename);
    else
      error("The GRID file you want to read: " + filename + ", cannot be found!");
    auto grid=GridBase::create(funcl, args, gridfile, g_min, g_max, g_nbins, sparsegrid, spline, true);
    if(grid->getDimension()!=args.size()) error("mismatch between dimensionality of input grid and number of arguments");
    for(unsigned i=0; i<args.size(); ++i)
      if( args[i]->isPeriodic()!=grid->getIsPeriodic()[i] ) error("periodicity mismatch between arguments and grid");
    gridfile.close();
    return grid;
}

} //namespace bias
} //namespace plumed

#endif
