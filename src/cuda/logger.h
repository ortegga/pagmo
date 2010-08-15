

#ifndef __PAGMO_CUDA_LOGGER__
#define __PAGMO_CUDA_LOGGER__

using namespace std;

class Logger : public ostream
{
 public:
  Logger ()
    {
      m_bActivated = true;
    }
  bool m_bActivated;
};

template <class T>
ostream & operator << (Logger & logger, const T & t)
{
  if (m_bActivated)
    {
      std::cout << t;
    }
  return logger;
}

Logger logger;

#endif
