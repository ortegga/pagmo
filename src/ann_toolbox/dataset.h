#ifndef __PAGMO_ANN_DATASET__
#define  __PAGMO_ANN_DATASET__


//TODOS
//This set of classes is supposed to wrap around the dataset for the neural networks
//Its supposed to abstract away the float format and the fact that CUDA is being 
//used or not from the neural network creation and use. 
namespace ann_toolbox {
  template <class T, class Container >
  class dataset 
  {
  public:
    //Random sources
    dataset(std::vector<T> & vec);
    dataset();
    dataset & operator = (const dataset & set);
    bool getValues(std::vector<T> & vec);
  private:
    Container<T> data;
  };

  template <class T>
  class ANNContainer
  {
  public:
    ANNContainer(std::vector<T> & vec);
    virtual ~ANNContainer();

  };

  template <class T>
  class VectorContainer : public ANNContainer<T>
  {
  public:
  VectorContainer(std::vector<T> & vec): data (vec){}
  private:
    std::vector<T> data;
  };
};

#endif
