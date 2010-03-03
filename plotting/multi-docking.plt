set nokey  # no legend
	set terminal png
	set output 'trajectory.png'
#	set terminal fig color big
#	set output 'trajectory.fig'
#set terminal postscript eps color "Helvetica" 12
#set output 'trajectory.eps'

# Plot the position of the spacecraft
set parametric
set xrange [-3.000000:3.000000]
set yrange [-3.000000:3.000000]
set size ratio 1.000000
set boxwidth 1 absolute
set style fill solid 1.0
set multiplot
# Plot the selected vicinity circle
plot [0:2*pi] 0.15*sin(t),0.15*cos(t) lt rgb "black"

# Plot the trajectories
plot 'multi_docking_1.dat' using 2:3 with line lt rgb "#FF0000" notitle
plot 'multi_docking_2.dat' using 2:3 with line lt rgb "#00FF00" notitle
plot 'multi_docking_3.dat' using 2:3 with line lt rgb "#0000FF" notitle
plot 'multi_docking_4.dat' using 2:3 with line lt rgb "#FF00FF" notitle
plot 'multi_docking_5.dat' using 2:3 with line lt rgb "#00FFFF" notitle


# Plot the thruster activation
# set size ratio .5 0.3
# set origin	0.5,0.04
# set yrange [-0.1:0.1]
# set xrange [0:9]
# set xtics  ""
# set ytics ("-1" -0.1, "0" 0.0, "1" 0.1)
# set title "Thrusters"
# plot 'multi_run_1.dat' using 1:8 with line notitle, 'multi_run_1.dat' using 1:9 with line notitle
# plot 'mutli_run_2.dat' using 1:8 with line notitle, 'mutli_run_2.dat' using 1:9 with line notitle
# plot 'mutli_run_3.dat' using 1:8 with line notitle, 'mutli_run_3.dat' using 1:9 with line notitle

#	# plot the rotation
#	set size ratio .5 0.3
#	set origin	0.5,0.68
#	set yrange [-2*pi:0]
#	set xrange [0:9]
#	set xtics  ""
#	set ytics ("0" 0, "Pi" pi, "2Pi" 2*pi)
#	set title "Orientation"
#	plot 'goodrun.dat' using 1:6 with line notitle

	#pause -1 "PRESS key to stop gnuplot!"
