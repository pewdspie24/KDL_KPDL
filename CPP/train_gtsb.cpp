#include "matrix/matrix.h"
#include <bits/stdc++.h>
using namespace std;
std::random_device randomDevice;
std::mt19937 engine(22);
double init_value(double factor){
    
    std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
    return (valueDistribution(engine) + 0.5)*factor;
}
class LinearLayer{
private:
    string name;
    string activation;
    int input_dim;
    int hidden_dim;
    Matrix weights;
    Matrix bias;
    std::vector<Matrix> nodes;

public:
    Matrix getWeights(){return this->weights;};
    Matrix getBias(){return this->bias;}
    void setWeights(Matrix weight){this->weights = weight;}
    void setBias(Matrix bias){this->bias=bias;};
    LinearLayer(string name, int input_dim, int hidden_dim){
        this->name = name;    
        this->input_dim = input_dim;
        this->hidden_dim=hidden_dim;
        this->weights = Matrix(input_dim, hidden_dim,-1);
        this->bias = Matrix(1,hidden_dim,log(1.));
        // init_param
        double factor = 2 * sqrt(6.0/(hidden_dim)) ;
        this->weights.Map([&factor](double x ){
            return init_value(factor);
        });

        nodes=std::vector<Matrix>(0);
    };
    Matrix forward(Matrix inputs_d, bool debug=false){
        Matrix inputs = Matrix(inputs_d.GetColumnVector(),inputs_d.GetHeight(), inputs_d.GetWidth());
        // inputs : batch_size, input_dim 
        Matrix outputs = inputs * this->weights + this->bias.repeat(inputs.GetHeight());
        nodes.clear();
        nodes.push_back(inputs.Map([](double x){return x;}));
        return outputs;
        
        
    }
    std::vector<Matrix> backward(Matrix previous_gradient_d){
        // rule backward: tra ve dao ham theo  list = (inputs, weight, bias)
        // previous_gradient = loss'(outputs )_(outputs) with outputs  = dau ra cua layer hien tai 
        // ->> previous_gradient  cos shape = shape cua dau ra = BS, hidden_dim 
        // goi inputs la dau vao cua layer hien tai de tinh loss' theo inputs ap dung qui tac chuoi
        // loss'(U(INPUTS)) = loss'(U) * U'(inputs')=previous_gradient * U'(inputs)
        // U(inputs) = inputs*weights + bias ->> U'(inputs) = Weights.T (hidden_dim, input_dim)
        // ->> loss'(inputs) = previous_gradient * Weight.T (bs, input_dim)
        Matrix previous_gradient=Matrix(previous_gradient_d.GetColumnVector(), previous_gradient_d.GetHeight(), previous_gradient_d.GetWidth());
        Matrix weight = Matrix(this->weights.GetColumnVector(), this->weights.GetHeight(), this->weights.GetWidth());
        Matrix delta_inputs = previous_gradient * Matrix::Transpose(weight); // bs,input_dim 
        
        // loss'(weight) = loss'(U) *U'(weight) voi U = inputs  * weights + bias 
        // U'(weights) = inputs.T :shape=input_dim,bs->> loss'(weight) = inputs.T * previous_gradient : shape = input_dim, hidden_dim 
        // inputs duoc luu vao nodes tai vi tri 0 luc forward
        Matrix delta_weight = this->nodes[0].Map([](double x){return x;}).Transpose() * previous_gradient;
        // delta_weight=delta_weight+this->weights*2.0*0.00001;
        // loss'(bias) = loss'(U) * U'(bias) voi U = inputs *weight + bias ->U'(bias) = 1
        // -> loss'(bias) = previous_gradient.sum(0) // vi bias luc compute da dc scale nen bs lan plus ->
        std::vector<double> data_bias = std::vector<double>(0);
        for(int col=0;col<previous_gradient.GetWidth();col++){
            double d=0;
            for(int row=0;row<previous_gradient.GetHeight();row++){
                d=d+previous_gradient.GetRow(row)[col];
            }
            data_bias.push_back(d);
        }
        if(data_bias.size()!=this->hidden_dim){
            cout<<"BUG BIAS"<<endl;
        }
        Matrix delta_bias = Matrix(data_bias, 1, this->hidden_dim);


        return std::vector<Matrix>({
            delta_inputs,
            delta_weight,
            delta_bias
        });
    }

};

class Activation{
private:
    std::string name;
    std::string typeActivation;
    std::vector<Matrix> nodes;
   
public:
    Activation(std::string name, std::string typeActivation){
        nodes = std::vector<Matrix>(0);
        this->name=name;
        this->typeActivation=typeActivation;
        
    }
    std::string getName(){
        return this->name;
    }
    std::string getTypeActivation(){
        return this->typeActivation;
    }
    Matrix forward(Matrix inputs_d){
        Matrix inputs = Matrix(inputs_d.GetColumnVector(),inputs_d.GetHeight(), inputs_d.GetWidth());
        Matrix activationOutputs;
        this->nodes.clear();
        if(this->typeActivation.compare("relu")==0){
            activationOutputs = inputs.Map([](double x){
                if(x<=0)return 0.;
                return x;
            });
            this->nodes.push_back(
                inputs.Map([](double x){return x;})
            );
          
        }
        else{
            if(this->typeActivation.compare("softmax")==0){
                std::vector<double> data_final = std::vector<double>(0);
                for(int row=0;row<inputs.GetHeight();row++){
                    double max = 0.0;
                    double sum = 0.0;
                    std::vector<double> data = inputs.GetRow((unsigned int)row);
                    for (int i = 0; i < data.size(); i++) if (max < data[i]) max = data[i];
                    for (int i = 0; i < data.size(); i++) {
                        data[i] = exp(data[i] - max);
                        sum += data[i];
                    }
                    for (int i = 0; i < data.size(); i++) 
                    {
                        data[i] /= sum;
                        if(data[i] > 1-1e-15)data[i] = 1-1e-15;
                        if(data[i] < 1e-15) data[i] = 1e-15;
                    }
                    for(auto &z:data)data_final.push_back(z);
                }
                // for(auto z:data_final)cout<<z<<" ";
                // cout<<"dsadsa"<<endl;
                activationOutputs = Matrix(data_final, inputs.m_Rows, inputs.m_Columns);

                // softmax can luu lai outputs de tiet kiem chi phi tinh toan
                this->nodes.push_back(
                    activationOutputs.Map([](double x){return x;})
                );
                // cout<<this->name<<" soft"<<" "<<inputs.m_Rows<<" "<<inputs.m_Columns<<endl;
                
            }
            if(this->typeActivation.compare("sigmoid")==0){
                activationOutputs=inputs.Map([](double x){
                    return 1./(1. + exp(-1*x));
                });
                this->nodes.push_back(
                    activationOutputs.Map([](double x){return x;})
                );
            }
            
        }
        return activationOutputs;
    }
    std::vector<Matrix> backward(Matrix previous_gradient_d){
        Matrix previous_gradient=Matrix(previous_gradient_d.GetColumnVector(), previous_gradient_d.GetHeight(), previous_gradient_d.GetWidth());
        if(this->typeActivation.compare("relu")==0){
            Matrix gradient_relu = Matrix(this->nodes[0]);
            gradient_relu=gradient_relu.Map([](double x){
                if(x<=0)return 0.;
                return 1.;
            });
            return std::vector<Matrix>({gradient_relu.ElementWise(previous_gradient)});
        }
        else{
            if(this->typeActivation.compare("softmax")==0 || this->typeActivation.compare("sigmoid")==0 ){
                Matrix gradient_inputs = this->nodes[0].Map(
                    [](double x){
                        return x*(1-x);
                    }
                );
                return std::vector<Matrix>({gradient_inputs.ElementWise(previous_gradient)});

            }
        }
    };
};
class CategoricalCrossentropy{
public:
    CategoricalCrossentropy(){};
    double compute(Matrix probMatrix, Matrix targetMatrix){
        // targetMatrix dang one-hot
        std::vector<double> predictionVector = probMatrix.GetColumnVector();
        std::vector<double> targetVector = targetMatrix.GetColumnVector();

        double sum = 0.0;
        std::vector<double>::iterator tIt = targetVector.begin();
        for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
        {
            double value_clip = *pIt;
           
            double value = -*tIt*log(value_clip);
            if (std::isinf(value) || std::isnan(value)) value = std::numeric_limits<int>::max()*1.0;
            sum += value;
        }
        return sum/probMatrix.GetHeight();
    };
    Matrix backward(Matrix& prediction, Matrix& target){
        auto z= (prediction - target) / double(prediction.GetHeight());
        auto m =  prediction.Map([](double x){
            return x*(1-x);
        });
        return z/m;
    };
};


class SgdOptimizer{
private:
    double lr;
    std::vector< std::vector<Matrix> > listOfParams;
public:
    SgdOptimizer(double lr = 0.1){
        this->lr =lr;
        listOfParams = std::vector< std::vector<Matrix> >(0);
    }
    void setLr(double lr){
        this->lr =lr;
    };
    double getLr(){return this->lr;};
    void step(std::vector< std::vector<Matrix> >  gradient_params, std::vector<LinearLayer * > & linears){
        for(int i = 0; i< linears.size(); i++){
            std::vector<Matrix> gradient_param = gradient_params[i];
            if(linears[i]->getWeights().GetHeight()!=gradient_param[1].GetHeight()){
                cout<<"BUGGGGGGGGGGGGGGGGG";
            }
            gradient_param[1]=gradient_param[1].Map([](double x){
                if(x>100){
                    x=100;
                }
                if(x<-100)x=-100;
                return x;
            });
            gradient_param[2]=gradient_param[2].Map([](double x){
                if(x>100){
                    x=100;
                }
                if(x<-100)x=-100;
                return x;
            });
            linears[i]->setWeights(
                linears[i]->getWeights() - this->lr * gradient_param[1]
            );
            linears[i]->setBias(
                linears[i]->getBias() - this->lr * gradient_param[2]
            );

        }

    }

};
std::vector<std::vector<double>> read_file(std::string path){
    string myText;

// Read from the text file
    ifstream MyReadFile(path);
    std::vector<string> a;
    std::string delim=",";
    std::vector<std::vector<double>> inputs;
    
    // Use a while loop together with the getline() function to read the file line by line
    while (getline (MyReadFile, myText)) {
        std::vector<double> input;
        auto start = 0U;
        auto end = myText.find(delim);
        // Matrix input = Matrix
        while (end != std::string::npos)
        {
            double value= std::stod(myText.substr(start, end - start));
            input.push_back(value);
            start = end + delim.length();
            end = myText.find(delim, start);
        }

        input.push_back (std::stod(myText.substr(start, end)));
        inputs.push_back(input);
    }
    MyReadFile.close();
    return inputs;

    
}
class AdamOptimzer{
private:
    std::vector<std::vector<Matrix>> listOfParamsv, listOfParamsw,listOfParamsp;
    double lr;
    double alpha1;
    double beta1;
    double beta2;
    double eps;
    double cache_beta1;
    double cache_beta2;
    int t;
public:
    AdamOptimzer(double lr){
        this->lr=lr;
        beta1=0.9;
        beta2=0.999;
        eps=1e-6;
        cache_beta1=1;
        cache_beta2=1;
        t=1;
    }
    void setLr(double lr){
        this->lr =lr;
    };
    double getLr(){return this->lr;};
    void step(std::vector< std::vector<Matrix> >  gradient_params, std::vector<LinearLayer * > & linears){
        if (listOfParamsv.size() != linears.size()){
            // note sgd ko can tracking gi ca vi param=param-lr*grad || de sau implement adam
            cout<<"Init params for optimizer tracking\n";
            listOfParamsv.clear();
            listOfParamsw.clear();
            for(auto &w_tracking: linears){
                std::vector<Matrix> params = std::vector<Matrix>({
                    w_tracking->getWeights().copy().Map([](double x){return 0;}),
                    w_tracking->getBias().copy().Map([](double x){return 0;})
                });
                listOfParamsv.push_back(params);
                listOfParamsw.push_back(std::vector<Matrix>({
                    w_tracking->getWeights().copy().Map([](double x){return 0;}),
                    w_tracking->getBias().copy().Map([](double x){return 0;})
                }));
                
            }
        }
        cache_beta1=cache_beta1*beta1;
        cache_beta2=cache_beta2*beta2;
        this->t=this->t+1;
        for(int i=0;i<linears.size();i++){
            std::vector<Matrix> gradient_param = gradient_params[i];
            gradient_param[1]=gradient_param[1].Map([](double x){
                if(x>200){
                    x=200;
                }
                if(x<-200)x=-200;
                return x;
            });
            gradient_param[2]=gradient_param[2].Map([](double x){
                if(x>200){
                    x=200;
                }
                if(x<-200)x=-200;
                return x;
            });
            listOfParamsv[i][0]= this->beta1 * listOfParamsv[i][0] + (1 - beta1) * gradient_param[1];
            listOfParamsv[i][1] = this->beta1 * listOfParamsv[i][1] + (1 - beta1) * gradient_param[2];

            listOfParamsw[i][0]= this->beta2 * listOfParamsw[i][0] + (1 - beta2) * gradient_param[1].copy().Map([](double x){return x*x;});
            listOfParamsw[i][1] = this->beta2 * listOfParamsw[i][1] + (1 - beta2) * gradient_param[2].copy().Map([](double x){return x*x;});
            
            auto v_bias_corr = listOfParamsv[i][0] / (1 - cache_beta1);
            auto s_bias_corr = listOfParamsw[i][0] / (1 - cache_beta2);
            double ep = eps;
            gradient_param[1]=(this->lr * v_bias_corr) / (s_bias_corr.copy().Map([&ep](double x){return (sqrt(x)+ep*1.0);}) );
            v_bias_corr = listOfParamsv[i][1] / (1 - cache_beta1);
            s_bias_corr = listOfParamsw[i][1] / (1 - cache_beta2);
            // double ep = eps;
            gradient_param[2]=(this->lr * v_bias_corr) / (s_bias_corr.copy().Map([&ep](double x){return (sqrt(x)+ep*1.0);}) );

            
            
            linears[i]->setWeights(
                linears[i]->getWeights()-gradient_param[1]
            );
            linears[i]->setBias(
                linears[i]->getBias() - gradient_param[2]
            );

        }
        

    }
};
Matrix feedForward(Matrix & inputs, std::vector<LinearLayer*> & linears, std::vector<Activation*> & activations){
    for(int i=0;i<linears.size();i++){
        inputs = linears[i]->forward(inputs);
        inputs = activations[i]->forward(inputs);        
    }
    return inputs;
};
std::vector< std::vector<Matrix> > feedBackward(Matrix & inputs, Matrix & targets, std::vector<LinearLayer*> & linears, std::vector<Activation*> & activations){
    int ns=targets.GetHeight();
    Matrix gradient = (inputs-targets).Map([&ns](double x){
        return x/double(ns);
    });
    std::vector<Matrix> global_gradient = std::vector<Matrix>({gradient});
    std::vector< std::vector<Matrix> >  tracking_gradient;
    for(int i=linears.size()-1;i>=0;i--){
        // cout<<gradient<<endl;
        if(i!=linears.size()-1){
            global_gradient=activations[i]->backward(gradient);
            gradient = global_gradient[0];
        }
        
        global_gradient=linears[i]->backward(gradient);
        gradient = global_gradient[0];
        
        tracking_gradient.push_back(
            std::vector<Matrix> ({global_gradient})
        );
        std::rotate(tracking_gradient.rbegin(), tracking_gradient.rbegin() + 1, tracking_gradient.rend());
    }

    return tracking_gradient;
}
int calculatorAccuracy(Matrix & prediction, Matrix & targets){
    int total_acc = 0;
    for(int row=0;row < prediction.GetHeight();row++){
        
        vector<double> data = prediction.GetRow(row);
        vector<double> tar = targets.GetRow(row);
        double value=data[0];
        int index_max=0;

        for(int i=1;i<data.size();i++){
            if(data[i] > value){
                value=data[i];
                index_max=i;
            }
        }

        double index_max_target=0;
        for(int i=1;i<tar.size();i++){
            if(tar[i]==1){
                index_max_target = i;
            }
        }
        if(index_max == index_max_target)total_acc =total_acc + 1;
        
    }
    return total_acc;
}
void save_model(std::vector<LinearLayer*> & linears, std::string path){
    std::ofstream outfile;
    outfile.open(path, std::ios::binary | std::ios::out);
    for(int i=0;i<linears.size();i++){
        linears[i]->getWeights().SaveMatrix(outfile);
        linears[i]->getBias().SaveMatrix(outfile);
    }
    outfile.close();
}
void load_model(std::vector<LinearLayer*> & linears, std::string path){
    std::ifstream infile;
    infile.open(path, std::ios::in | std::ios::binary);
    for(int i=0;i<linears.size();i++){
        Matrix weights = Matrix::LoadMatrix(infile);
        Matrix bias = Matrix::LoadMatrix(infile);
        linears[i]->setWeights(weights);
        linears[i]->setBias(bias);
    }
}
int main(){
    int num_class = 7;
    std::vector<int> hidden_dims= std::vector<int>({16,32,7});
    int input_dim = 32*32*3;
    std::vector<LinearLayer*> linears;
    std::vector<Activation*> activations;
    CategoricalCrossentropy loss = CategoricalCrossentropy();
    auto optim = AdamOptimzer(0.01);
    for(int i=0;i<hidden_dims.size(); i++){
        int input=0;
        if(i==0){
            input = input_dim;
        }
        else{
            input=hidden_dims[i-1];
        }
        linears.push_back(
            new LinearLayer("dense_"+std::to_string(i+1),input, hidden_dims[i])
        );
        if(i!=hidden_dims.size()-1){
            activations.push_back(
                new Activation("activation_relu_"+std::to_string(i+1),"relu")
            );
        }
        else{
            activations.push_back(new Activation("activation_softmax","softmax"));
        }
    }

    cout<<"Init model done\n";
    // cout<<linears[0]->getWeights()<<endl<<linears[0]->getBias()<<endl;
    auto train_ds=read_file("/mnt/01D4B61792FFD5D0/btl_co_chi/src/datasets/gtsb/train.csv");
    auto test_ds = read_file("/mnt/01D4B61792FFD5D0/btl_co_chi/src/datasets/gtsb/test.csv");
   
    cout<<"Read data done\n";
    
    vector<int> indexs;
    for(int i=0;i<train_ds.size();i++)indexs.push_back(i);
    double best_acc = 0;
    std::string checkpoint="./ckpts/gtsb.ckpt";
    std::random_shuffle(indexs.begin(), indexs.end());
    for(int epoch=0; epoch<100; epoch++){
        cout<<"Training at epoch "<<to_string(epoch)<<endl;
        if(epoch == 30)optim.setLr(optim.getLr()/2.);
        if(epoch == 50)optim.setLr(optim.getLr()/2.);
        // if(epoch == 80)optim.setLr(optim.getLr()/2.);
        
        double loss_total=0;
        int acc_total=0;
        std::random_shuffle(indexs.begin(), indexs.end()); 
        for (int batch_start=0;;batch_start = batch_start + 64){
            int batch_end = batch_start + 64;
            if(batch_end > train_ds.size())batch_end = train_ds.size();
            if(batch_start >= batch_end )break;

            vector<double> inputs_data=vector<double>(0);
            vector<double> inputs_target=vector<double>(0);
            
            for(int index=batch_start;index<batch_end;index++){
                int index_in_ds = indexs[index];
                for(int i=0;i<input_dim;i++){
                    inputs_data.push_back(
                        train_ds[index_in_ds][i]
                    );
                }
                for(int i=0;i<num_class;i++){
                    if(train_ds[index_in_ds][input_dim] == i+1){
                         inputs_target.push_back(1.);
                    }
                    else{
                        inputs_target.push_back(0);
                    }
                }
            }
            Matrix inputs = Matrix(inputs_data, batch_end-batch_start, input_dim);
            Matrix targets = Matrix(inputs_target, batch_end-batch_start,num_class);
            auto prediction = feedForward(inputs, linears, activations);
            double loss_step =  loss.compute(prediction, targets);
            int acc_step = calculatorAccuracy(prediction, targets);
            loss_total = loss_total + loss_step;
            acc_total = acc_total + acc_step;
            auto tracking_gradient = feedBackward(prediction, targets, linears, activations);
            optim.step(
                tracking_gradient, linears
            );
            if(batch_start >= batch_end || batch_start >= train_ds.size() || batch_end >= train_ds.size())break;
  

        }
        cout<<"total loss "<<loss_total/train_ds.size()<<endl<<"acc total "<<double(acc_total)/train_ds.size()<<endl;
        // evaluation
        vector<double> inputs_data=vector<double>(0);
        vector<double> inputs_target=vector<double>(0);
        for(int index=0;index<test_ds.size();index++){
            
            for(int i=0;i<input_dim;i++){
                inputs_data.push_back(
                    test_ds[index][i]
                );
            }
            for(int i=0;i<num_class;i++){
                if(test_ds[index][input_dim] == i+1){
                    inputs_target.push_back(1);
                }
                else{
                    inputs_target.push_back(0);
                }
            }
            
            
        }
        Matrix inputs = Matrix(inputs_data, test_ds.size(), input_dim);
        Matrix targets = Matrix(inputs_target, test_ds.size(),num_class);


        //forward
        auto prediction = feedForward(inputs, linears, activations);
        int total_acc = calculatorAccuracy(prediction, targets);
        cout<<"-------------------------------------\n";
        cout<<"Validation \n";
        cout<<double(total_acc)/test_ds.size()<<endl;
        cout<<"--------------------------------------\n"; 
        if(double(total_acc)/test_ds.size() > best_acc ){
            best_acc=double(total_acc)/test_ds.size();
            save_model(linears,checkpoint);
            cout<<"Save checkpoint"<<endl;
        }
    }
    cout<<"Training end\n";
    load_model(linears,checkpoint);
    cout<<"Load model best done\n";
    cout<<"Produce results\n";
    vector<double> inputs_data=vector<double>(0);
    vector<double> inputs_target=vector<double>(0);
    for(int index=0;index<test_ds.size();index++){
        
        for(int i=0;i<input_dim;i++){
            inputs_data.push_back(
                test_ds[index][i]
            );
        }
        for(int i=0;i<num_class;i++){
            if(test_ds[index][input_dim] == i+1){
                inputs_target.push_back(1);
            }
            else{
                inputs_target.push_back(0);
            }
        }
        
        
    }
    Matrix inputs = Matrix(inputs_data, test_ds.size(), input_dim);
    Matrix targets = Matrix(inputs_target, test_ds.size(),num_class);
    //forward
    auto prediction = feedForward(inputs, linears, activations);
    int total_acc = calculatorAccuracy(prediction, targets);
    cout<<"Acc = "<<double(total_acc)/test_ds.size()<<endl;
}