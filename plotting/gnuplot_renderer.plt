set nokey  # no legend
	set terminal png
	set output 'trajectory.png'
#	set terminal fig color big
#	set output 'trajectory.fig'
#set terminal postscript eps color "Helvetica" 12
#set output 'trajectory.eps'


# Plot the position of the spacecraft
set xrange [-8.000000:8.000000]
set yrange [-8.000000:8.000000]
set size ratio 1.000000
set boxwidth 1 absolute
set style fill solid 1.0
set multiplot

plot 'goodrun.dat' using 2:4 with line notitle

# Plot the thruster activation
set size ratio .5 0.3
set origin	0.5,0.04
set yrange [-0.1:0.1]
set xrange [0:9]
set xtics  ""
set ytics ("-1" 0, "0" .5, "1" 1)
set title "Thrusters"
plot 'goodrun.dat' using 1:6 with line notitle, 'goodrun.dat' using 1:7 with line notitle

# plot the rotation
set size ratio .5 0.3
set origin	0.5,0.68
set yrange [0:2*pi]
set xrange [0:9]
set xtics  ""
set ytics ("0" 0, "Pi" pi, "2Pi" 2*pi)
set title "Orientation"
plot 'goodrun.dat' using 1:5 with line notitle

#pause -1 "PRESS key to stop gnuplot!"
