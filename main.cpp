/*****************************************************************************
 *   Copyright (C) 2004-2013 The PaGMO development team,                     *
 *   Advanced Concepts Team (ACT), European Space Agency (ESA)               *
 *   http://apps.sourceforge.net/mediawiki/pagmo                             *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Developers  *
 *   http://apps.sourceforge.net/mediawiki/pagmo/index.php?title=Credits     *
 *   act@esa.int                                                             *
 *                                                                           *
 *   This program is free software; you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation; either version 2 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program; if not, write to the                           *
 *   Free Software Foundation, Inc.,                                         *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.               *
 *****************************************************************************/

#include <iostream>
#include <iomanip>
#include "src/pagmo.h"

using namespace pagmo;

// Example in C++ of the use of PaGMO 1.1.5

int main()
{
	/*
	//We instantiate the problem Schwefel with diemnsion 50
	pagmo::problem::schwefel prob(50);
	//We instantiate the algorithm differential evolution with 500 generations
	pagmo::algorithm::de algo(3000);

	//1 - Evolution takes place on the same thread as main
	//We instantiate a population containing 20 candidate solutions to the Schwefel problem
	pagmo::population pop(prob,20);
	algo.evolve(pop);
	
	std::cout << "Evolve method of the algorithm: " << pop.champion().f << std::endl; 
	
	//2 - Evolution takes place on a separate thread
	//We instantiate an island containing 20 candidate solutions to the Schwefel problem
	pagmo::island isl(algo,prob,20);
	isl.evolve();
	
	std::cout << "Evolve method of the island: " << isl.get_population().champion().f << std::endl; 

	//3 - 8 Evolutions take place in parallel on 8 separte islands containing, each, 20
	// candidate solutions to the Schwefel problem
	pagmo::archipelago archi(algo,prob,8,20);
	archi.evolve();

	std::vector<double> temp;
	for (archipelago::size_type i = 0; i < archi.get_size(); ++i) {
		temp.push_back(archi.get_island(i)->get_population().champion().f[0]);
	}
	std::cout << "Evolve method of the archipelago: " << *std::min_element(temp.begin(),temp.end()) << std::endl; 
	
	//4 - 8 Evolutions take place in parallel on 8 separte islands with migration
	pagmo::algorithm::de algo2(300);
	pagmo::topology::one_way_ring topo;
	pagmo::archipelago archi2(algo2,prob,8,20,topo);
	archi2.evolve(10);
	
	temp.clear();
	for (archipelago::size_type i = 0; i < archi.get_size(); ++i) {
		temp.push_back(archi2.get_island(i)->get_population().champion().f[0]);
	}
	std::cout << "Evolve method of the archipelago (with migration): " << *std::min_element(temp.begin(),temp.end()) << std::endl; 
	return 0;
	*/

	pagmo::problem::cec2006 cec(1);
	pagmo::algorithm::cstrs_immune_system algo(pagmo::algorithm::jde(1), pagmo::algorithm::jde(10), 500, pagmo::algorithm::cstrs_immune_system::INFEASIBILITY,	pagmo::algorithm::cstrs_immune_system::BEST25,
						pagmo::algorithm::cstrs_immune_system::HAMMING,0.9,0.9);

	pagmo::population pop(cec);
	double x1[13] = {0.3858577217775121, 0.065554407219352129, 0.43299067179761508, 0.70631990525484056, 0.304159924475222, 0.65933800559837508, 0.88864980322693299, 0.39185283746992638, 0.88390032687783915, 26.518721388998401, 45.207762420351827, 53.182203557738106, 0.33742527812286127};
	double x2[13] = {0.6324883636378047, 0.0087498932497460657, 0.13658314901499224, 0.87175772333483437, 0.87354183167210309, 0.20269411281343608, 0.1167553718493366, 0.85804067463158518, 0.59615041160193982, 50.647870838159648, 12.902131876167999, 79.860534579564302, 0.57846767856385739};
	double x3[13] = {0.60703138517013144, 0.21358332424959059, 0.35912124063756323, 0.47519713898003246, 0.41954302125538945, 0.26615549214771761, 0.41962516784466786, 0.40851630240077696, 0.57605869068100191, 79.410683316545416, 66.466450521825493, 9.6166455200201284, 0.27698968085477915};
	double x4[13] = {0.85017266497394672, 0.16975106853286448, 0.70517872530167836, 0.24074455352427293, 0.46488189752306042, 0.67683123295492109, 0.4600740840024713, 0.24718834610797913, 0.43803249454910542, 54.451928403112859, 15.639572798919232, 37.310881649457883, 0.20038858148683758};
	double x5[13] = {0.8276092993981834, 0.54846903306353312, 0.73983105927789339, 0.55901697869796863, 0.57662255420772368, 0.27739881241489073, 0.095545154812381128, 0.67344892769286702, 0.33211508949003132, 74.451228325744623, 25.76484485052859, 24.75344230473091, 0.034308505799828026};
	double x6[13] = {0.21694594712537807, 0.28148557531557472, 0.19433155984979322, 0.56497624383664657, 0.27718046462474533, 0.59800104693325906, 0.090384566885823858, 0.92589974153819554, 0.93671407497343395, 14.916530638015146, 33.623374182389298, 78.525345755446807, 0.20032939127561988};
	double x7[13] = {0.64341349033383466, 0.19594272666122592, 0.60362721082969273, 0.87536303610478683, 0.57270167029907881, 0.73659620026768025, 0.46616277282511476, 0.072759044816333329, 0.2021389873703896, 17.241415789408165, 30.336571061449291, 71.458555104074151, 0.88845031392004969};
	double x8[13] = {0.9594197643758875, 0.23636678515518028, 0.47783419201357091, 0.14963227908080512, 0.94795459814031702, 0.42801495959175284, 0.55802780591205803, 0.64922248174120512, 0.24882968410623008, 53.803035648571296, 8.6280134241274453, 17.136534706593309, 0.19996052518787266};
	double x9[13] = {0.25663894611305693, 0.1909568837494362, 0.68142138992059031, 0.24172779772654351, 0.37061451954179603, 0.11410070928424787, 0.42031311727685505, 0.87253605885201324, 0.014594661515001661, 43.262351194384507, 42.819493288898869, 11.588564392699752, 0.90997628885421733};
	double x10[13] = {0.94322785731456094, 0.014353260868624318, 0.35726151951257634, 0.56188228325128264, 0.83149089633660012, 0.60223976430537363, 0.49912519548806955, 0.67786125186652768, 0.44221745387532607, 31.225082733769582, 63.899805350927963, 89.713187374244185, 0.38320116497953904};
	double x11[13] = {0.13812960588313317, 0.79886840140744297, 0.45899587861983449, 0.66002068955299364, 0.83930318366040169, 0.56888180324253312, 0.57658117741297588, 0.90137274161498837, 0.85043161021496871, 65.996810404934791, 22.364752279041866, 72.105902428169344, 0.94389677328451782};
	double x12[13] = {0.054293450684323119, 0.66208108229155727, 0.27867747560927825, 0.24110408568749264, 0.13435923949736406, 0.96965800468594665, 0.57310748943190148, 0.99099489456969536, 0.99599660117941013, 30.717310032908429, 44.255612901661223, 24.018498542080025, 0.41330546511835564};
	double x13[13] = {0.7505546751915162, 0.71805582631896669, 0.57377046643358653, 0.25126165936444522, 0.43871795483681453, 0.59923481120818067, 0.32525528910320389, 0.37931473357723178, 0.58614526208049256, 51.631651507366882, 79.157317167296881, 92.279843153948704, 0.21131680432961986};
	double x14[13] = {0.034934083942602001, 0.51725212959008005, 0.21590566414364432, 0.89570697372450425, 0.86120555434524348, 0.48030415980397478, 0.98002244389566684, 0.73080221897661701, 0.64105700395244014, 51.867014856005511, 61.060022643694012, 74.749378386731991, 0.89599402442882337};
	double x15[13] = {0.24609893689970264, 0.60936909645025494, 0.086897639264954307, 0.9831797893763401, 0.049447371315896049, 0.65379113175334069, 0.55962683456747797, 0.50983752984997466, 0.063249313726498002, 74.012712223455424, 72.367273866530724, 64.519486232502388, 0.97932084791100493};
	double x16[13] = {0.076848618999335372, 0.89135399459402365, 0.79149806901993358, 0.12252919092031433, 0.22735373315199325, 0.6828350339589484, 0.59934544646187149, 0.79351154818358083, 0.56247138601521129, 45.982521103940854, 94.577981732794214, 1.8881344660020005, 0.9416281638490922};
	double x17[13] = {0.2495284298291125, 0.33436147493171475, 0.39279983218586523, 0.98091844256224192, 0.023727444795639485, 0.24292630139983018, 0.024473893203818875, 0.41836376267650621, 0.5680644164100066, 30.538208304752956, 40.046005740071422, 69.097577210602523, 0.60311172208197306};
	double x18[13] = {0.6514067898238558, 0.9684991521207813, 0.68459720363051346, 0.18291518840414867, 0.8231146455978795, 0.083342622863387561, 0.38392400583767028, 0.84018211462382908, 0.42564464547903569, 6.6551778895828306, 4.1639547807076127, 91.677604056495099, 0.99130736700461952};
	double x19[13] = {0.81296598684208732, 0.97048760313749938, 0.14140413521921502, 0.63249230724460048, 0.60050726134185695, 0.56011942443583962, 0.61908687274424423, 0.72024112754731462, 0.64445361011615176, 92.954965921893518, 70.672975664499305, 58.727779105106848, 0.16015535000328995};
	double x20[13] = {0.96002691140828489, 0.38008205954171359, 0.040507717279155031, 0.096988277794775257, 0.82850322203283255, 0.026500348606447233, 0.44215086481043642, 0.6442398781831109, 0.65348547209360319, 62.2901658833797, 50.510767273046042, 63.271171071900767, 0.6633580644346857};

	decision_vector xx1(13);std::copy(x1,x1 + 13,xx1.begin()); pop.push_back(xx1);
	decision_vector xx2(13);std::copy(x2,x2 + 13,xx2.begin()); pop.push_back(xx2);
	decision_vector xx3(13);std::copy(x3,x3 + 13,xx3.begin()); pop.push_back(xx3);
	decision_vector xx4(13);std::copy(x4,x4 + 13,xx4.begin()); pop.push_back(xx4);
	decision_vector xx5(13);std::copy(x5,x5 + 13,xx5.begin()); pop.push_back(xx5);
	decision_vector xx6(13);std::copy(x6,x6 + 13,xx6.begin()); pop.push_back(xx6);
	decision_vector xx7(13);std::copy(x7,x7 + 13,xx7.begin()); pop.push_back(xx7);
	decision_vector xx8(13);std::copy(x8,x8 + 13,xx8.begin()); pop.push_back(xx8);
	decision_vector xx9(13);std::copy(x9,x9 + 13,xx9.begin()); pop.push_back(xx9);
	decision_vector xx10(13);std::copy(x10,x10 + 13,xx10.begin()); pop.push_back(xx10);
	decision_vector xx11(13);std::copy(x11,x11 + 13,xx11.begin()); pop.push_back(xx11);
	decision_vector xx12(13);std::copy(x12,x12 + 13,xx12.begin()); pop.push_back(xx12);
	decision_vector xx13(13);std::copy(x13,x13 + 13,xx13.begin()); pop.push_back(xx13);
	decision_vector xx14(13);std::copy(x14,x14 + 13,xx14.begin()); pop.push_back(xx14);
	decision_vector xx15(13);std::copy(x15,x15 + 13,xx15.begin()); pop.push_back(xx15);
	decision_vector xx16(13);std::copy(x16,x16 + 13,xx16.begin()); pop.push_back(xx16);
	decision_vector xx17(13);std::copy(x17,x17 + 13,xx17.begin()); pop.push_back(xx17);
	decision_vector xx18(13);std::copy(x18,x18 + 13,xx18.begin()); pop.push_back(xx18);
	decision_vector xx19(13);std::copy(x19,x19 + 13,xx19.begin()); pop.push_back(xx19);
	decision_vector xx20(13);std::copy(x20,x20 + 13,xx20.begin()); pop.push_back(xx20);

	algo.evolve(pop);
	std::cout<< pop.champion().f[0]<<std::endl;

	std::cout<<std::endl;

	algo.evolve(pop);
	std::cout<< pop.champion().f[0]<<std::endl;

	int uscita;
	std::cin>>uscita;
}
