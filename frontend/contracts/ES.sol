// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract ES {

    enum UserType {
        Administrator,
        Voter
    }

    struct Constituency {
        string stateCode;
        string constituencyType;
        string constituencyNumber;
        string constituencyName;
    }

    struct Voter {
        string name;
        string fatherName;
        string imageUrl;
        string gender;
        string dob;
        string uid;
        string constituency;
        string location;
        string phoneNumber;
        bool isVoted;
    }

    struct Candidate {
        string name;
        string fatherName;
        string imageUrl;
        string gender;
        string dob;
        string uid;
        string constituency;
        string location;
        string politicalAffiliation;
        string phoneNumber;
        uint votes;
    }

    mapping(string => string[]) public irisSamples;
    mapping(string => Voter) private voters;
    mapping(string => Candidate) private candidates;
    mapping(string => string[]) private constituenciesVoters;
    mapping(string => string[]) private constituenciesCandidates;
    mapping(string => Constituency) public constituencies;
    string[] public constituencyIds;
    string[] private  parties;
    mapping(string => uint256) private partyIndex;
    UserType private userType;


    constructor() {
        if (msg.sender == 0x212e25Fe820C221C089Cd1163917FF1eE81eBa84) {
            userType = UserType.Administrator;
        } else {
            userType = UserType.Voter;
        }
    }

    modifier onlyAdministrator() {
        require(
            userType == UserType.Administrator,
            "Only administrators can call this function"
        );
        _;
    }

    function existsConstituency(
        string memory _id
    ) public view returns (bool) {
        return bytes(constituencies[_id].constituencyName).length > 0;
    }

    function registerConstituency(
        string memory _stateCode,
        string memory _cType,
        string memory _cNumber,
        string memory _cName 
    ) public onlyAdministrator {
        string memory Id = string(abi.encodePacked(_stateCode, "-", _cType, "-", _cNumber, "-", _cName));

        // require(
        //     bytes(constituencies[Id].constituencyName).length == 0, 
        //     "Constituency is already exists");

        constituencies[Id] = Constituency(
            _stateCode,
            _cType,
            _cNumber,
            _cName
        );

        constituencyIds.push(Id);
    }

    function getConstituencies()
    public view returns (Constituency[] memory){
        Constituency[] memory constituencyList = new Constituency[](constituencyIds.length);
        for (uint i = 0; i < constituencyIds.length; i++) {
            constituencyList[i] = constituencies[constituencyIds[i]];
        }
        return constituencyList;
    }

    function existsVoter(
        string memory _uid
    ) public view returns (bool) {
        return bytes(voters[_uid].name).length > 0;
    }

    function registerVoter(
        string memory _name,
        string memory _fatherName,
        string memory _imageUrl,
        string memory _gender,
        string memory _dob,
        string memory _uid,
        string memory _constituency,
        string memory _location,
        string memory _phoneNumber,
        string[] memory _irisSamples
    ) public onlyAdministrator {

        // require(
        //     bytes(voters[_uid].name).length == 0,
        //     "Voter with this UID is already exists"
        // );

        // require(
        //     bytes(constituencies[_constituency].constituencyName).length != 0, 
        //     "Constituency is not exists. Select a valid contituency.");

        Voter memory voter = Voter(
            _name,
            _fatherName,
            _imageUrl,
            _gender,
            _dob,
            _uid,
            _constituency,
            _location,
            _phoneNumber,
            false
        );

        voters[_uid] = voter;
        constituenciesVoters[_constituency].push(_uid);
        irisSamples[_uid] = _irisSamples;
    }

    function existsCandidate(
        string memory _uid
    ) public view returns (bool) {
        return bytes(candidates[_uid].name).length > 0;
    }

    function registerCandidate(
        string memory _name,
        string memory _fatherName,
        string memory _imageUrl,
        string memory _gender,
        string memory _dob,
        string memory _uid,
        string memory _constituency,
        string memory _location,
        string memory _politicalAffiliation,
        string memory _phoneNumber
    ) public onlyAdministrator {

        // require(
        //     bytes(candidates[_uid].name).length == 0,
        //     "Candidate with this UID is already exists"
        // );

        // require(
        //     bytes(constituencies[_constituency].constituencyName).length != 0, 
        //     "Constituency is not exists. Select a valid contituency.");

        // require(checkParty(_constituency, _politicalAffiliation),
        // "Multiple candidates can't register for a same party in the same constituency");

        Candidate memory candidate = Candidate(
            _name,
            _fatherName,
            _imageUrl,
            _gender,
            _dob,
            _uid,
            _constituency,
            _location,
            _politicalAffiliation,
            _phoneNumber,
            0
        );
        candidates[_uid] = candidate;
        constituenciesCandidates[_constituency].push(_uid);
    }

    function getCandidateIds(string memory _constituency)
        public
        view
        returns (string[] memory)
    {
        uint256 candidateCount = constituenciesCandidates[_constituency].length;
        string[] memory candidateIds = new string[](candidateCount);
        for (uint256 i = 0; i < candidateCount; i++) {
            candidateIds[i] = constituenciesCandidates[_constituency][i];
        }
        return candidateIds;
    }

    function getVoterIds(string memory _constituency)
        public
        view
        returns (string[] memory)
    {
        uint256 voterCount = constituenciesVoters[_constituency].length;
        string[] memory voterIds = new string[](voterCount);
        for (uint256 i = 0; i < voterCount; i++) {
            voterIds[i] = constituenciesVoters[_constituency][i];
        }
        return voterIds;
    }

    function isVoterEligible(string memory _uid) public view returns (bool) 
    {
        require(bytes(voters[_uid].name).length != 0, "Voter is not registered");

        if(voters[_uid].isVoted)
            return false;
        else
            return true;
    }

    function getVoter(string memory _uid)
        public
        view
        returns (Voter memory)
    {
        require(bytes(voters[_uid].name).length != 0, "Voter is not registered");
        return voters[_uid];
    }

    function getCandidate(string memory _uid)
        public
        view
        returns (Candidate memory)
    {
        require(bytes(candidates[_uid].name).length != 0, "Candidate is not registered");
        return candidates[_uid];
    }

    function getIrisSample(string memory _uid)
    public
    view
    returns (string[] memory _irisSample)
    {
        require(bytes(voters[_uid].name).length != 0, "Voter is not registered");
        return irisSamples[_uid];
    }

    function vote(
        string memory _voter_uid,
        string memory _candidate_uid
    ) public {
        // require(bytes(voters[_voter_uid].name).length != 0, "Voter is not registered");

        // require(!voters[_voter_uid].isVoted, "You are already voted");

            candidates[_candidate_uid].votes++;
            voters[_voter_uid].isVoted = true;
    }


    function checkParty(string memory _constituency, string memory _politicalAffiliation) view public returns (bool) {
        bool check = true;
        for(uint i = 0; i < constituenciesCandidates[_constituency].length; i++) {
            if(keccak256(abi.encodePacked(candidates[constituenciesCandidates[_constituency][i]].politicalAffiliation)) == keccak256(abi.encodePacked(_politicalAffiliation)))
                check = false;
        }
        return check;
    }

}
