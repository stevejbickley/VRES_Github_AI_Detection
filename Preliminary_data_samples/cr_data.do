/* Examining the Github Data */ 

clear all 
set more on
capture log close
*---------------------------------------------------------------* 
local me = c(username)
//cap cd "C:\"
cd "C:\Users/`me'\Dropbox\GITHUB_CritMass\example_csv"

fs *.csv

foreach f in `r(files)' { 
 
* Taking a random GitHub file. 
import delimited "`f'", clear

*---------------------------------------------------------------*
* Readjusting V2 and V3 to the right columns
replace v2 = trim(v2)
replace v3 = trim(v3) 

// Adjustments due to spelling differences 
replace v3 = v2 if v2 == "1 deletion(-)" 
replace v2 = "" if v2 == "1 deletion(-)" 

forvalues i = 0(1)100{
 replace v3 = v2 if v2 == "`i' deletions(-)"
 replace v2 = "" if v2 == "`i' deletions(-)"
 //di `i'
}

*---------------------------------------------------------------*
* Splitting the variables for specific identification of information
split v1, p(";") gen(new_) 
split v2, p(" ") gen(insert_) 
split v3, p(" ") gen(delete_)

drop insert_2 delete_2
destring insert_1 delete_1 new_5, replace

*---------------------------------------------------------------*
* Rename the variables and add labels 
rename insert_1 insertions
label variable insertions "Number of line insertions to project" 

rename delete_1 deletions 
label variable deletions " Number of line deletions from projet" 
// is this the deletions of line of code or the 'digits' as some deletions occur 
// within a short period of time from the same author

rename new_3 contributor 
label variable contributor "Person contributing to the project"

rename new_4 date_contribution 
label variable date_contribution "Date of the contribution"

rename  new_1 repository
label variable repository "Name of the repository; log"
/*
rename new_5 
label variable new_5 

rename new_2 
label variable new_2
*/
*---------------------------------------------------------------*
* Constructing the date variables 
* remove the -0400 
split date_contribution, p(" -" " +") gen(date_) 

* Generating and formatting the recognised date variable 
gen double eventtime = clock(date_1, "#MDhmsY")
format eventtime %tc

//It may be interesting to examing the day of the week most work was conducted. 
//So we adjust to get the day using the calendar component. 

gen contribution_dow = dow(dofc(eventtime))
label variable contribution_dow "Day of the week of the contribution"
label define dow 0 "Sun" 1 "Mon" 2 "Tues" 3 "Wed" 4 "Thur" 5 "Fri" ///
 6 "Sat"  
label values contribution_dow dow

*Keep the data file tidy, remove non-used variables 
//drop date_1 date_2
*---------------------------------------------------------------*

* Setting the cd in PhD folder for temp and normal file save. Dropobx limitations
save "\\qut.edu.au\Documents\StaffHome\StaffGroupM$\moyn\Desktop\PhD\Github/`f'.dta", replace 
}
* Set new directory for the .dta files 
cd "\\qut.edu.au\Documents\StaffHome\StaffGroupM$\moyn\Desktop\PhD\Github"
fs *.dta

* Append all sample files 
foreach f in `r(files)' { 
append using "`f'" 
}
*---------------------------------------------------------------*
* Encode contributors & repository 
encode contributor, gen(contributor_id) 
label variable contributor_id "Identifier for contributors"

encode repository, gen(repository_id) 
label variable repository_id "Identifier for the repository" 
 
*---------------------------------------------------------------*
* Drop variables not used, keep orignial 3 variables 
drop repository new_2 date_contribution contributor new_5 date_1 date_2 

*---------------------------------------------------------------*
save cr_combined, replace 

